# maske list should be hard coded to [148, 198, 232, 256, 274, 290, 306, 320]

class ProgressiveMask(nn.Module):
    def __init__(self, mask_list):
        super(ProgressiveMask, self).__init__()
        if not mask_list:
            raise ValueError("mask_list cannot be empty.")
        self.mask_list = mask_list

    def forward(self, x, mask_under, iter_idx):
        # Use the appropriate mask based on the iteration index
        pred_kspace = fft2c(rearrange(x, 'b c h w -> b h w c'))

        mask_size = self.mask_list[min(iter_idx, len(self.mask_list) - 1)]
        cy, cx = pred_kspace.shape[1] // 2, pred_kspace.shape[2] // 2
        half = mask_size // 2
        mask = torch.zeros_like(pred_kspace)
        mask[:, cy-half:cy+half, cx-half:cx+half, :] = 1.0

        # combine with the original under-sampling mask
        combined_mask = torch.logical_or(mask.bool(), mask_under.bool()).float()

        # Apply the combined mask to the predicted k-space
        masked_kspace = pred_kspace * combined_mask

        updated_img = ifft2c(masked_kspace)
        updated_img = rearrange(updated_img, 'b h w c -> b c h w')
        return updated_img
    

class DataConsistencyLayer(nn.Module):
    def __init__(self):  # 去掉固定的 dc_weight
        super(DataConsistencyLayer, self).__init__()
        # 改为可学习的参数，并初始化为 1
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(self, pred_img, mask_under, lq_kspace):
        pred_kspace = fft2c(rearrange(pred_img, 'b c h w -> b h w c'))

        # 使用 torch.where 更安全
        zero = torch.zeros_like(pred_kspace)
        soft_dc = torch.where(mask_under.bool(), pred_kspace - lq_kspace, zero) * self.dc_weight
        updated_kspace = pred_kspace - soft_dc

        updated_img = ifft2c(updated_kspace)
        updated_img = rearrange(updated_img, 'b h w c -> b c h w')

        return updated_img