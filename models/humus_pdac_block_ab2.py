import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_

from models.humus_pdac_block import (
    ConvBlock,
    DownsampConvBlock,
    PatchEmbed,
    PatchEmbedLearned,
    PatchExpandSkip,
    PatchMerging,
    PatchUnEmbed,
    PatchUnEmbedLearned,
    SwinTransformerBlock,
    TransposeConvBlock,
)


class BasicLayerNoCondition(nn.Module):
    """Swin Transformer layer without AdaLN or external condition input."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(lambda tensor: blk(tensor, x_size), x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RSTBNoCondition(nn.Module):
    """Residual Swin Transformer block without condition injection."""

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
        block_type="B",
    ):
        super().__init__()
        conv_dim = dim // (patch_size**2)
        divide_out_ch = 1
        self.input_resolution = input_resolution

        self.residual_group = BasicLayerNoCondition(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(conv_dim, conv_dim // divide_out_ch, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(conv_dim // 4, conv_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(conv_dim // 4, conv_dim // divide_out_ch, 3, 1, 1),
            )
        else:
            raise ValueError("Unknown residual connection type.")

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=dim,
            norm_layer=None,
        )

        self.block_type = block_type
        if block_type == "B":
            self.reshape = torch.nn.Identity()
        elif block_type == "D":
            self.reshape = PatchMerging(input_resolution, dim)
        elif block_type == "U":
            self.reshape = PatchExpandSkip([res // 2 for res in input_resolution], dim * 2)
        else:
            raise ValueError("Unknown RSTB block type.")

    def forward(self, x, x_size, skip=None):
        if self.block_type == "U":
            if skip is None:
                raise ValueError("Skip connection is required for patch expand.")
            x = self.reshape(x, skip)

        out = self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))
        block_out = self.patch_embed(out) + x

        if self.block_type == "D":
            block_out = (self.reshape(block_out), block_out)

        return block_out


class HUMUSBlockNoCondition(nn.Module):
    """HUMUS block with a plain backbone and no condition injection."""

    def __init__(
        self,
        img_size,
        in_chans,
        patch_size=1,
        embed_dim=66,
        depths=None,
        num_heads=None,
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        img_range=1.0,
        resi_connection="1conv",
        bottleneck_depth=2,
        bottleneck_heads=24,
        conv_downsample_first=True,
        out_chans=None,
        no_residual_learning=False,
        **kwargs,
    ):
        super().__init__()
        depths = [2, 2, 2] if depths is None else depths
        num_heads = [3, 6, 12] if num_heads is None else num_heads

        num_out_ch = in_chans if out_chans is None else out_chans
        self.center_slice_out = out_chans == 1
        self.img_range = img_range
        self.register_buffer("mean", torch.zeros(1, 1, 1, 1))
        self.window_size = window_size
        self.conv_downsample_first = conv_downsample_first
        self.no_residual_learning = no_residual_learning

        input_conv_dim = embed_dim
        self.conv_first = nn.Conv2d(
            in_chans,
            input_conv_dim // 2 if self.conv_downsample_first else input_conv_dim,
            3,
            1,
            1,
        )

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**self.num_layers)
        self.mlp_ratio = mlp_ratio

        if self.conv_downsample_first:
            img_size = [im // 2 for im in img_size]
            self.conv_down_block = ConvBlock(input_conv_dim // 2, input_conv_dim, 0.0)
            self.conv_down = DownsampConvBlock(input_conv_dim, input_conv_dim)

        if patch_size > 1:
            self.patch_embed = PatchEmbedLearned(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=embed_dim,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,
            )
        patches_resolution = self.patch_embed.patches_resolution

        if patch_size > 1:
            self.patch_unembed = PatchUnEmbedLearned(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=embed_dim,
                out_chans=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None,
            )
        else:
            self.patch_unembed = PatchUnEmbed(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )

        if self.ape:
            num_patches = self.patch_embed.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim_scaler = 2**i_layer
            self.layers_down.append(
                RSTBNoCondition(
                    dim=int(embed_dim * dim_scaler),
                    input_resolution=(
                        patches_resolution[0] // dim_scaler,
                        patches_resolution[1] // dim_scaler,
                    ),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=False,
                    img_size=[im // dim_scaler for im in img_size],
                    patch_size=1,
                    resi_connection=resi_connection,
                    block_type="D",
                )
            )

        dim_scaler = 2**self.num_layers
        self.layer_bottleneck = RSTBNoCondition(
            dim=int(embed_dim * dim_scaler),
            input_resolution=(
                patches_resolution[0] // dim_scaler,
                patches_resolution[1] // dim_scaler,
            ),
            depth=bottleneck_depth,
            num_heads=bottleneck_heads,
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            img_size=[im // dim_scaler for im in img_size],
            patch_size=1,
            resi_connection=resi_connection,
            block_type="B",
        )

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim_scaler = 2 ** (self.num_layers - i_layer - 1)
            source_layer = self.num_layers - 1 - i_layer
            self.layers_up.append(
                RSTBNoCondition(
                    dim=int(embed_dim * dim_scaler),
                    input_resolution=(
                        patches_resolution[0] // dim_scaler,
                        patches_resolution[1] // dim_scaler,
                    ),
                    depth=depths[source_layer],
                    num_heads=num_heads[source_layer],
                    window_size=window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:source_layer]) : sum(depths[: source_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=False,
                    img_size=[im // dim_scaler for im in img_size],
                    patch_size=1,
                    resi_connection=resi_connection,
                    block_type="U",
                )
            )

        self.norm_down = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(input_conv_dim, input_conv_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(input_conv_dim, input_conv_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(input_conv_dim // 4, input_conv_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(input_conv_dim // 4, input_conv_dim, 3, 1, 1),
            )
        else:
            raise ValueError("Unknown residual connection type.")

        if self.conv_downsample_first:
            self.conv_up_block = ConvBlock(input_conv_dim, input_conv_dim // 2, 0.0)
            self.conv_up = TransposeConvBlock(input_conv_dim, input_conv_dim // 2)

        self.conv_last = nn.Conv2d(
            input_conv_dim // 2 if self.conv_downsample_first else input_conv_dim,
            num_out_ch,
            3,
            1,
            1,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def check_image_size(self, x):
        _, _, height, width = x.size()

        mod_pad_h = (self.window_size - height % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - width % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")

        total_downsamp = int(2 ** (self.num_layers - 1))
        pad_h = (total_downsamp - height % total_downsamp) % total_downsamp
        pad_w = (total_downsamp - width % total_downsamp) % total_downsamp
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        skip_cons = []
        for layer in self.layers_down:
            x, skip = layer(x, x_size=layer.input_resolution)
            skip_cons.append(skip)
        x = self.norm_down(x)

        x = self.layer_bottleneck(x, self.layer_bottleneck.input_resolution)

        for i, layer in enumerate(self.layers_up):
            x = layer(x, x_size=layer.input_resolution, skip=skip_cons[-i - 1])
        x = self.norm_up(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x):
        channels, height, width = x.shape[1:]
        center_slice = (channels - 1) // 2
        x = self.check_image_size(x)

        mean = self.mean.to(device=x.device, dtype=x.dtype)
        x = (x - mean) * self.img_range

        if self.conv_downsample_first:
            x_first = self.conv_first(x)
            x_down = self.conv_down(self.conv_down_block(x_first))
            res = self.conv_after_body(self.forward_features(x_down))
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
            res = self.conv_after_body(self.forward_features(x_first)) + x_first

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
