import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.humus_net_ab1 import HUMUSNetAB1
from models.humus_net_ab2 import HUMUSNetAB2
from models.humus_net_ab3 import HUMUSNetAB3


def _synthetic_multicoil_batch():
    torch.manual_seed(0)
    kspace = torch.randn(1, 2, 16, 16, 2)
    mask = torch.zeros(1, 1, 1, 16, 1, dtype=torch.uint8)
    mask[:, :, :, 6:10, :] = 1
    return kspace * mask, mask


def _shared_model_kwargs(num_list, embed_dim=66):
    return {
        "num_list": num_list,
        "num_cascades": len(num_list),
        "sens_chans": 4,
        "sens_pools": 1,
        "mask_center": False,
        "num_adj_slices": 1,
        "img_size": [16, 16],
        "patch_size": 1,
        "window_size": 4,
        "embed_dim": embed_dim,
        "depths": [1],
        "num_heads": [3],
        "mlp_ratio": 2.0,
        "bottleneck_depth": 1,
        "bottleneck_heads": 6,
        "resi_connection": "1conv",
        "conv_downsample_first": True,
        "use_checkpoint": False,
        "no_residual_learning": False,
    }


class TestStanfordAblationModels(unittest.TestCase):
    def _assert_forward_shapes(self, model):
        masked_kspace, mask = _synthetic_multicoil_batch()
        model.eval()
        with torch.no_grad():
            output, kspace_set, inter_mask, inter_prob = model(masked_kspace, mask)

        self.assertEqual(tuple(output.shape), (1, 16, 16))
        self.assertEqual(tuple(kspace_set.shape[-4:]), (2, 16, 16, 2))
        self.assertEqual(tuple(inter_mask.shape[-4:]), (1, 1, 16, 1))
        self.assertEqual(tuple(inter_prob.shape[-1:]), (16,))

    def test_ab1_forward_uses_full_mask_schedule(self):
        model = HUMUSNetAB1(**_shared_model_kwargs([16, 16], embed_dim=66))
        self._assert_forward_shapes(model)

        _, mask = _synthetic_multicoil_batch()
        masked_kspace, _ = _synthetic_multicoil_batch()
        with torch.no_grad():
            _, _, _, inter_prob = model(masked_kspace, mask)
        self.assertTrue(bool(torch.all(inter_prob == 1)))

    def test_ab2_forward_keeps_progressive_mask_projection(self):
        model = HUMUSNetAB2(**_shared_model_kwargs([8, 16], embed_dim=12))
        self._assert_forward_shapes(model)

        _, mask = _synthetic_multicoil_batch()
        masked_kspace, _ = _synthetic_multicoil_batch()
        with torch.no_grad():
            _, _, _, inter_prob = model(masked_kspace, mask)
        self.assertLess(float(inter_prob[:, 0].sum()), float(inter_prob[:, 1].sum()))

    def test_ab3_forward_repeats_only_original_sampling_mask(self):
        model = HUMUSNetAB3(**_shared_model_kwargs([16, 16], embed_dim=12))
        self._assert_forward_shapes(model)

        _, mask = _synthetic_multicoil_batch()
        masked_kspace, _ = _synthetic_multicoil_batch()
        with torch.no_grad():
            _, _, _, inter_prob = model(masked_kspace, mask)
        self.assertTrue(bool(torch.equal(inter_prob[:, 0], inter_prob[:, 1])))
        self.assertEqual(float(inter_prob[:, 0].sum()), float(mask.sum()))


if __name__ == "__main__":
    unittest.main()
