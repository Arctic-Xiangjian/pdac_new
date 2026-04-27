import torch
import torch.nn as nn

from models.humus_pdac_block import HUMUSBlock_pdac


class HUMUSBlockStageCondition(HUMUSBlock_pdac):
    """HUMUS block with cascade-stage conditioning instead of mask conditioning."""

    def __init__(self, *args, num_cascades: int = 8, embed_dim: int = 66, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        self.mask_embedder = None
        self.stage_embedder = nn.Embedding(num_cascades, embed_dim)

    def _stage_condition(self, stage_index, batch_size: int, device: torch.device) -> torch.Tensor:
        if not torch.is_tensor(stage_index):
            stage_index = torch.tensor(stage_index, device=device, dtype=torch.long)
        else:
            stage_index = stage_index.to(device=device, dtype=torch.long)

        if stage_index.ndim == 0:
            stage_index = stage_index.expand(batch_size)
        elif stage_index.numel() == 1 and batch_size != 1:
            stage_index = stage_index.reshape(1).expand(batch_size)
        else:
            stage_index = stage_index.reshape(batch_size)

        return self.stage_embedder(stage_index)

    def forward(self, x, stage_index):
        c, height, width = x.shape[1:]
        center_slice = (c - 1) // 2
        x = self.check_image_size(x)

        t = self._stage_condition(stage_index, x.shape[0], x.device)

        mean = self.mean.to(device=x.device, dtype=x.dtype)
        x = (x - mean) * self.img_range

        if self.conv_downsample_first:
            x_first = self.conv_first(x)
            x_down = self.conv_down(self.conv_down_block(x_first))
            res = self.conv_after_body(self.forward_features(x_down, t))
            res = self.conv_up(res)
            res = torch.cat([res, x_first], dim=1)
            res = self.conv_up_block(res)

            res = self.conv_last(res)

            if self.no_residual_learning:
                x = res
            else:
                if self.center_slice_out:
                    x = x[:, center_slice, ...].unsqueeze(1)
                x = x + res
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, t)) + x_first

            if self.no_residual_learning:
                x = self.conv_last(res)
            else:
                if self.center_slice_out:
                    x = x[:, center_slice, ...].unsqueeze(1)
                x = x + self.conv_last(res)

        if self.center_slice_out:
            x = x / self.img_range + self.mean[:, center_slice, ...].unsqueeze(1)
        else:
            x = x / self.img_range + self.mean

        return x[:, :, :height, :width]
