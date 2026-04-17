from typing import Sequence

import torch


def validate_center_mask_schedule(
    num_list: Sequence[int], num_cascades: int, img_size: Sequence[int]
) -> Sequence[int]:
    if img_size is None or len(img_size) < 2:
        raise ValueError("img_size must contain the target height and width.")

    schedule = [int(mask_size) for mask_size in num_list]
    if len(schedule) != num_cascades:
        raise ValueError(
            f"Expected num_list to have {num_cascades} entries, got {len(schedule)}."
        )
    if any(mask_size <= 0 for mask_size in schedule):
        raise ValueError("All center-mask sizes must be positive.")
    if any(curr > nxt for curr, nxt in zip(schedule, schedule[1:])):
        raise ValueError("Center-mask schedule must be non-decreasing.")

    final_mask_size = min(int(img_size[0]), int(img_size[1]))
    if any(mask_size > final_mask_size for mask_size in schedule):
        raise ValueError(
            f"Center-mask sizes cannot exceed the target resolution {final_mask_size}."
        )
    if schedule[-1] != final_mask_size:
        raise ValueError(
            f"The final center-mask size must be {final_mask_size}, got {schedule[-1]}."
        )

    return schedule


def expand_line_mask(mask: torch.Tensor, height: int) -> torch.Tensor:
    if mask.shape[2] == height:
        return mask
    return mask.expand(-1, -1, height, -1, -1)


def mask_to_width_profile(mask: torch.Tensor) -> torch.Tensor:
    return mask.amax(dim=2).squeeze(1).squeeze(-1)


def width_profile_to_mask(profile: torch.Tensor) -> torch.Tensor:
    return profile[:, None, None, :, None]


def build_effective_center_mask(
    base_mask: torch.Tensor,
    height: int,
    width: int,
    mask_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask_size = min(int(mask_size), height, width)
    start_y = (height - mask_size) // 2
    start_x = (width - mask_size) // 2

    square_mask = torch.zeros(
        base_mask.shape[0],
        1,
        height,
        width,
        1,
        device=device,
        dtype=dtype,
    )
    square_mask[:, :, start_y : start_y + mask_size, start_x : start_x + mask_size, :] = 1

    base_mask_2d = expand_line_mask(base_mask.to(device=device, dtype=dtype), height)
    return torch.logical_or(square_mask.bool(), base_mask_2d.bool()).to(dtype=dtype)
