import torch

from models.humus_net_ab1 import HUMUSNetAB1
from pl_modules.mri_module import MriModule
from pl_modules.pdac_module import PDACModule


class PDACModuleAB1(PDACModule):
    """Lightning module for Stanford2D ablation 1."""

    def __init__(
        self,
        lr: float = 0.0001,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        max_epoch: int = 50,
        num_adj_slices: int = 1,
        mask_center: bool = False,
        use_compile: bool = False,
        logger_type="tb",
        **kwargs,
    ):
        if "num_log_images" in kwargs:
            num_log_images = kwargs["num_log_images"]
            kwargs.pop("num_log_images", None)
        else:
            num_log_images = 16

        MriModule.__init__(self, num_log_images)
        self.save_hyperparameters()

        self.logger_type = logger_type
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.use_compile = use_compile
        self.model = HUMUSNetAB1(
            num_adj_slices=num_adj_slices,
            mask_center=mask_center,
            **kwargs,
        )
        self.loss = torch.nn.L1Loss()

    @staticmethod
    def add_model_specific_args(parent_parser):
        return PDACModule.add_model_specific_args(parent_parser)
