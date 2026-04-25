import json
import sys
import tempfile
import unittest
from pathlib import Path

import fastmri
import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fastmri_data import SliceDataset
from data.stanford.stanford_data import StanfordSliceDataset
from models.center_mask_scheduler import validate_center_mask_schedule
from utils.fixed_kspace_hierarchy_multi_coil_common import (
    DEFAULT_REPRESENTATION,
    RAW_COIL_REPRESENTATION,
    _accumulate_shell_gram_inplace,
    _build_kspace_representation,
    _center_gram_inplace,
    _compute_window_gram_direct,
    infer_max_available_kspace_coils,
    _materialize_window_gram,
    run_hierarchy_job,
)
from utils.power import generate_progressive_windows, parse_info_file


FASTMRI_ROOT = Path("/v/ai/nobackup/arctic/public_lowlevel/data/fastMRI/knee/multicoil_train")
STANFORD_ROOT = Path("/working2/arctic/Recon/stanford_convert")


def _complex_mode(height: int, width: int, freq_x: float, freq_y: float) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    real = torch.cos(np.pi * freq_x * x) + 0.15 * freq_y * y
    imag = torch.sin(np.pi * freq_y * y) - 0.1 * freq_x * x
    return torch.complex(real.float(), imag.float())


def _make_multicoil_kspace(num_coils: int, height: int = 16, width: int = 16) -> torch.Tensor:
    modes = [
        _complex_mode(height, width, 1.0, 0.5),
        _complex_mode(height, width, 2.0, 1.5),
        _complex_mode(height, width, 3.0, 2.5),
    ]
    weights = torch.tensor(
        [
            [4.0, 0.3, 0.0],
            [3.0, 0.2, 0.0],
            [0.8, 1.4, 0.2],
            [0.1, 0.4, 1.1],
            [0.0, 0.1, 0.7],
        ],
        dtype=torch.float32,
    )
    phase = torch.exp(
        1j * torch.tensor([0.0, 0.2, -0.3, 0.4, -0.5], dtype=torch.float32)
    )

    coil_images = []
    for coil_idx in range(num_coils):
        mix = (
            weights[coil_idx, 0] * modes[0]
            + weights[coil_idx, 1] * modes[1]
            + weights[coil_idx, 2] * modes[2]
        )
        coil_images.append(mix * phase[coil_idx])

    coil_images = torch.stack(coil_images, dim=0).to(torch.complex64)
    return fastmri.fft2c(torch.view_as_real(coil_images))


class TestVirtualCoilRepresentation(unittest.TestCase):
    def test_virtual_coil_representation_has_fixed_shape_and_zero_padding(self):
        raw_kspace = torch.view_as_complex(_make_multicoil_kspace(3).contiguous())

        represented = _build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
        )

        self.assertEqual(tuple(represented.shape), (4, 16, 16, 2))
        self.assertTrue(
            bool(torch.allclose(represented[3], torch.zeros_like(represented[3]), atol=1e-6))
        )

    def test_virtual_coil_components_are_sorted_by_energy(self):
        raw_kspace = torch.view_as_complex(_make_multicoil_kspace(4).contiguous())

        represented = _build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
        )

        energies = represented.square().sum(dim=(1, 2, 3))
        self.assertGreater(float(energies[0]), float(energies[1]))
        self.assertGreater(float(energies[1]), float(energies[2]))

    def test_virtual_coil_phase_anchoring_is_deterministic(self):
        raw_kspace = torch.view_as_complex(_make_multicoil_kspace(4).contiguous())

        represented_a = _build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
        )
        represented_b = _build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
        )

        self.assertTrue(bool(torch.allclose(represented_a, represented_b, atol=1e-6)))


class TestRawCoilRepresentation(unittest.TestCase):
    def test_raw_coil_representation_preserves_coil_order_and_pads(self):
        raw_kspace = _make_multicoil_kspace(3)

        represented = _build_kspace_representation(
            torch.view_as_complex(raw_kspace.contiguous()),
            target=None,
            uniform_train_resolution=(16, 16),
            representation=RAW_COIL_REPRESENTATION,
            num_virtual_coils=5,
            calibration_window=8,
        )

        self.assertEqual(tuple(represented.shape), (5, 16, 16, 2))
        self.assertTrue(bool(torch.allclose(represented[:3], raw_kspace, atol=1e-5)))
        self.assertTrue(
            bool(torch.allclose(represented[3:], torch.zeros_like(represented[3:]), atol=1e-6))
        )


class TestMaxAvailableCoilInference(unittest.TestCase):
    def test_stanford_dataset_uses_largest_hdf5_coil_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for fname, num_coils in (("a.h5", 3), ("b.h5", 5)):
                with h5py.File(root / fname, "w") as hf:
                    hf.create_dataset(
                        "kspace",
                        data=np.zeros((2, num_coils, 8, 8), dtype=np.complex64),
                    )
                    hf.create_dataset(
                        "reconstruction_rss",
                        data=np.zeros((2, 8, 8), dtype=np.float32),
                    )

            dataset = StanfordSliceDataset(
                root=root,
                data_partition="train",
                train_val_split=1.0,
                train_val_seed=0,
                transform=None,
            )

            self.assertEqual(infer_max_available_kspace_coils(dataset), 5)

    def test_adjacent_slices_expand_effective_coil_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with h5py.File(root / "sample.h5", "w") as hf:
                hf.create_dataset(
                    "kspace",
                    data=np.zeros((4, 3, 8, 8), dtype=np.complex64),
                )
                hf.create_dataset(
                    "reconstruction_rss",
                    data=np.zeros((4, 8, 8), dtype=np.float32),
                )

            dataset = StanfordSliceDataset(
                root=root,
                data_partition="train",
                train_val_split=1.0,
                train_val_seed=0,
                transform=None,
                num_adj_slices=3,
            )

            self.assertEqual(infer_max_available_kspace_coils(dataset), 9)


class TestShellIncrementalGram(unittest.TestCase):
    def test_shell_incremental_gram_matches_direct_window_gram(self):
        rng = np.random.default_rng(0)
        cache_np = rng.standard_normal((4, 4, 8, 8, 2)).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_path = tmpdir_path / "cache.dat"
            raw_gram_path = tmpdir_path / "raw_gram.dat"

            cache_mm = np.memmap(
                cache_path,
                mode="w+",
                dtype=np.float32,
                shape=cache_np.shape,
            )
            cache_mm[:] = cache_np
            cache_mm.flush()

            cumulative_gram = np.memmap(
                raw_gram_path,
                mode="w+",
                dtype=np.float64,
                shape=(cache_np.shape[0], cache_np.shape[0]),
            )
            cumulative_gram[:] = 0.0
            cumulative_gram.flush()
            cumulative_norm_sq = np.zeros(cache_np.shape[0], dtype=np.float64)

            for window_size in (2, 4, 6, 8):
                _accumulate_shell_gram_inplace(
                    cache_mm,
                    cumulative_gram=cumulative_gram,
                    cumulative_norm_sq=cumulative_norm_sq,
                    window_size=window_size,
                    block_size=2,
                )
                gram_path = tmpdir_path / f"gram_{window_size}.dat"
                incremental = _materialize_window_gram(
                    cumulative_gram,
                    cumulative_norm_sq,
                    gram_path,
                    block_size=2,
                    normalize_per_sample=True,
                )
                incremental_centered = np.asarray(
                    _center_gram_inplace(incremental, block_size=2),
                    dtype=np.float64,
                )

                direct = _compute_window_gram_direct(
                    cache_np,
                    window_size,
                    block_size=2,
                    normalize_per_sample=True,
                )
                direct_centered = np.asarray(
                    _center_gram_inplace(direct.copy(), block_size=2),
                    dtype=np.float64,
                )

                np.testing.assert_allclose(
                    incremental_centered,
                    direct_centered,
                    atol=1e-8,
                    rtol=1e-6,
                )


class _HierarchySmokeMixin:
    def _assert_hierarchy_job_smoke(self, dataset, dataset_tag: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            result = run_hierarchy_job(
                dataset=dataset,
                dataset_tag=dataset_tag,
                output_prefix=tmpdir_path / f"{dataset_tag}_raw",
                tmp_dir=tmpdir_path / "scratch",
                num_samples=4,
                uniform_train_resolution=(32, 32),
                normalize_per_sample=True,
                representation=DEFAULT_REPRESENTATION,
                num_virtual_coils=4,
                calibration_window=32,
                metadata={
                    "source": "raw_kspace",
                    "representation": DEFAULT_REPRESENTATION,
                    "num_virtual_coils": 4,
                    "calibration_window": 32,
                },
            )

            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertIn("windows", payload)
            self.assertIn("infos", payload)
            self.assertIn("delta_infos", payload)
            self.assertIn("r_eff_by_window", payload)
            self.assertEqual(payload["metadata"]["source"], "raw_kspace")
            self.assertEqual(payload["metadata"]["representation"], DEFAULT_REPRESENTATION)
            self.assertEqual(payload["metadata"]["num_virtual_coils"], 4)
            self.assertEqual(payload["metadata"]["calibration_window"], 32)

            infos = np.asarray(payload["infos"], dtype=np.float64)
            self.assertTrue(bool(np.all(np.diff(infos) >= -1e-10)))

            info_by_window = parse_info_file(Path(result["json_path"]))
            schedule = generate_progressive_windows(
                info_by_window,
                acs_size=16,
                num_blocks=8,
                shell_power=0.25,
                log_alpha=20.0,
                img_size=32,
            )
            validate_center_mask_schedule(schedule, 8, [32, 32])


class TestFastMRIHierarchySmoke(unittest.TestCase, _HierarchySmokeMixin):
    @unittest.skipUnless(FASTMRI_ROOT.exists(), "fastMRI multi-coil dataset is unavailable")
    def test_fastmri_multicoil_hierarchy_smoke(self):
        dataset = SliceDataset(
            root=FASTMRI_ROOT,
            challenge="multicoil",
            transform=None,
        )
        self._assert_hierarchy_job_smoke(dataset, "fastmri_smoke")


class TestStanfordHierarchySmoke(unittest.TestCase, _HierarchySmokeMixin):
    @unittest.skipUnless(STANFORD_ROOT.exists(), "Stanford multi-coil dataset is unavailable")
    def test_stanford_multicoil_hierarchy_smoke(self):
        dataset = StanfordSliceDataset(
            root=STANFORD_ROOT,
            data_partition="train",
            train_val_split=0.8,
            train_val_seed=0,
            transform=None,
        )
        self._assert_hierarchy_job_smoke(dataset, "stanford_smoke")
