import fastmri
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from models.center_mask_scheduler import (
    expand_line_mask,
    mask_to_width_profile,
    width_profile_to_mask,
)
from models.humus_net_ab2 import VarNetBlockNoCondition
from models.humus_net_pdac import SensitivityModel
from models.humus_pdac_block_ab2 import HUMUSBlockNoCondition


class HUMUSNetAB3(nn.Module):
    """Stanford2D ablation 3: plain HUMUS/VarNet backbone without condition or mask projection."""

    def __init__(
        self,
        num_list=None,
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
                VarNetBlockNoCondition(
                    HUMUSBlockNoCondition(
                        in_chans=2 * self.num_adj_slices,
                        **kwargs,
                    ),
                    self.num_adj_slices,
                    im_size,
                )
                for _ in range(num_cascades)
            ]
        )
        self.num_cascades = num_cascades

    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor):
        center_slice = (self.num_adj_slices - 1) // 2

        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()
        initial_prob = mask_to_width_profile(
            expand_line_mask(mask.to(masked_kspace), masked_kspace.shape[-3])
        )

        kspace_out_set = []
        inter_mask_set = []
        inter_prob_set = []
        for cascade in self.cascades:
            if self.use_checkpoint:
                kspace_pred = checkpoint.checkpoint(
                    cascade,
                    kspace_pred,
                    masked_kspace,
                    mask,
                    sens_maps,
                )
            else:
                kspace_pred = cascade(
                    kspace_pred,
                    masked_kspace,
                    mask,
                    sens_maps,
                )

            kspace_out_set.append(kspace_pred)
            inter_mask_set.append(width_profile_to_mask(initial_prob).to(kspace_pred.dtype))
            inter_prob_set.append(initial_prob)

            kspace_pred = torch.chunk(kspace_pred, self.num_adj_slices, dim=1)[center_slice]

        out = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

        inter_mask_set = torch.stack(inter_mask_set, dim=1)
        inter_prob_set = torch.stack(inter_prob_set, dim=1)
        kspace_out_set = torch.stack(kspace_out_set, dim=1)
        return out, kspace_out_set, inter_mask_set, inter_prob_set
