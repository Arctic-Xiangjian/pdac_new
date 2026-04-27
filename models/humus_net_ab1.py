from typing import Tuple

import fastmri
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.center_mask_scheduler import (
    build_effective_center_mask,
    mask_to_width_profile,
    validate_center_mask_schedule,
    width_profile_to_mask,
)
from models.humus_net_pdac import SensitivityModel
from models.humus_pdac_block_ab1 import HUMUSBlockStageCondition


class HUMUSNetAB1(nn.Module):
    """Stanford2D ablation 1: stage conditioning with full-size mask projection."""

    def __init__(
        self,
        num_list,
        num_cascades: int = 8,
        sens_chans: int = 16,
        sens_pools: int = 4,
        mask_center: bool = True,
        num_adj_slices: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.use_checkpoint = kwargs["use_checkpoint"]
        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        im_size = kwargs["img_size"]
        self.cascades = nn.ModuleList(
            [
                VarNetBlockStageCondition(
                    HUMUSBlockStageCondition(
                        in_chans=2 * self.num_adj_slices,
                        num_cascades=num_cascades,
                        **kwargs,
                    ),
                    self.num_adj_slices,
                    im_size,
                )
                for _ in range(num_cascades)
            ]
        )
        full_mask_size = min(int(im_size[0]), int(im_size[1]))
        self.progressive_mask_schedule = validate_center_mask_schedule(
            [full_mask_size] * num_cascades, num_cascades, im_size
        )
        self.num_cascades = num_cascades

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor):
        center_slice = (self.num_adj_slices - 1) // 2

        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        kspace_out_set = []
        inter_mask_set = []
        inter_prob_set = []
        for stage_idx, cascade in enumerate(self.cascades):
            stage_index = torch.full(
                (kspace_pred.shape[0],),
                stage_idx,
                device=kspace_pred.device,
                dtype=torch.long,
            )
            if self.use_checkpoint:
                kspace_pred = checkpoint.checkpoint(
                    cascade,
                    kspace_pred,
                    masked_kspace,
                    mask,
                    sens_maps,
                    stage_index,
                )
            else:
                kspace_pred = cascade(
                    kspace_pred,
                    masked_kspace,
                    mask,
                    sens_maps,
                    stage_index,
                )

            current_mask = build_effective_center_mask(
                mask,
                kspace_pred.shape[-3],
                kspace_pred.shape[-2],
                self.progressive_mask_schedule[stage_idx],
                device=kspace_pred.device,
                dtype=kspace_pred.dtype,
            )
            kspace_pred = kspace_pred * current_mask
            current_prob = mask_to_width_profile(current_mask)
            kspace_out_set.append(kspace_pred)

            kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[center_slice]

            inter_mask_set.append(width_profile_to_mask(current_prob).to(current_mask.dtype))
            inter_prob_set.append(current_prob)

        out = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

        inter_mask_set = torch.stack(inter_mask_set, dim=1)
        inter_prob_set = torch.stack(inter_prob_set, dim=1)
        kspace_out_set = torch.stack(kspace_out_set, dim=1)
        return out, kspace_out_set, inter_mask_set, inter_prob_set


class VarNetBlockStageCondition(nn.Module):
    def __init__(self, model: nn.Module, num_adj_slices=1, im_size=None):
        super().__init__()
        self.num_adj_slices = num_adj_slices
        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))
        self.im_size = im_size

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        _, channels, _, _, _ = sens_maps.shape
        return fastmri.fft2c(
            fastmri.complex_mul(
                x.repeat_interleave(channels // self.num_adj_slices, dim=1),
                sens_maps,
            )
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width, _ = x.shape
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).view(
            batch_size,
            self.num_adj_slices,
            channels // self.num_adj_slices,
            height,
            width,
            2,
        ).sum(dim=2, keepdim=False)

    @staticmethod
    def complex_to_chan_dim(x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width, two = x.shape
        if two != 2:
            raise ValueError("Expected complex dimension of size 2.")
        return x.permute(0, 4, 1, 2, 3).reshape(batch_size, 2 * channels, height, width)

    @staticmethod
    def chan_complex_to_last_dim(x: torch.Tensor) -> torch.Tensor:
        batch_size, channels2, height, width = x.shape
        if channels2 % 2 != 0:
            raise ValueError("Expected an even channel count.")
        channels = channels2 // 2
        return x.view(batch_size, 2, channels, height, width).permute(
            0, 2, 3, 4, 1
        ).contiguous()

    @staticmethod
    def norm(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, height * width)
        mean = x.mean(dim=2).view(batch_size, channels, 1, 1)
        std = x.std(dim=2).view(batch_size, channels, 1, 1)
        x = x.view(batch_size, channels, height, width)
        return (x - mean) / std, mean, std

    @staticmethod
    def unnorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    def pad_width(self, x):
        pad_w = self.im_size[1] - x.shape[-1]
        if pad_w > 0:
            pad_w_left = pad_w // 2
            pad_w_right = pad_w - pad_w_left
        else:
            pad_w_left = pad_w_right = 0
        return (
            torch.nn.functional.pad(
                x, (pad_w_left, pad_w_right, 0, 0), "reflect"
            ),
            pad_w_left,
            pad_w_right,
        )

    def unpad_width(self, x, pad_w_left, pad_w_right):
        if pad_w_left > 0:
            x = x[:, :, :, pad_w_left:]
        if pad_w_right > 0:
            x = x[:, :, :, :-pad_w_right]
        return x

    def apply_model(self, x, stage_index):
        x = self.complex_to_chan_dim(x)
        if self.im_size is not None:
            x, pad_left, pad_right = self.pad_width(x)
        x, mean, std = self.norm(x)

        x = self.model(x, stage_index)

        x = self.unnorm(x, mean, std)
        if self.im_size is not None:
            x = self.unpad_width(x, pad_left, pad_right)
        return self.chan_complex_to_last_dim(x)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        stage_index: torch.Tensor,
    ) -> torch.Tensor:
        zero = current_kspace.new_zeros(1, 1, 1, 1, 1)

        restore_im = self.apply_model(
            self.sens_reduce(current_kspace, sens_maps), stage_index
        )
        model_term = self.sens_expand(restore_im, sens_maps)
        soft_dc = torch.where(mask.bool(), model_term - ref_kspace, zero) * self.dc_weight
        return model_term - soft_dc
