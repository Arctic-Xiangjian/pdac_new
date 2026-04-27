import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import fastmri
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.fastmri_data import SliceDataset
from data.stanford.stanford_data import StanfordSliceDataset
from models.center_mask_scheduler import validate_center_mask_schedule
from utils import fixed_kspace_hierarchy_multi_coil_common as cpu_common
from utils import fixed_kspace_hierarchy_multi_coil_common_gpu as gpu_common
from utils.power import generate_progressive_windows, parse_info_file


CUDA_AVAILABLE = torch.cuda.is_available()
CUDA_DEVICE = "cuda:0"
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


class _VariableCoilDataset:
    def __init__(self, coil_counts, height: int = 16, width: int = 16):
        self.height = height
        self.width = width
        self.samples = [
            torch.view_as_complex(
                _make_multicoil_kspace(num_coils, height=height, width=width).contiguous()
            )
            for num_coils in coil_counts
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        target = np.zeros((self.height, self.width), dtype=np.float32)
        return self.samples[idx], None, target, {}, f"synthetic_{idx}.h5", idx


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
class TestGPUVirtualCoilRepresentation(unittest.TestCase):
    def test_gpu_representation_has_fixed_shape_and_zero_padding(self):
        raw_kspace = torch.view_as_complex(_make_multicoil_kspace(3).contiguous())

        represented = gpu_common._build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=gpu_common.DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
            device=CUDA_DEVICE,
        )

        self.assertEqual(tuple(represented.shape), (4, 16, 16, 2))
        self.assertTrue(
            bool(
                torch.allclose(
                    represented[3],
                    torch.zeros_like(represented[3]),
                    atol=1e-6,
                )
            )
        )

    def test_gpu_representation_matches_cpu_on_synthetic_data(self):
        raw_kspace = torch.view_as_complex(_make_multicoil_kspace(4).contiguous())

        cpu_repr = cpu_common._build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=cpu_common.DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
        )
        gpu_repr = gpu_common._build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=gpu_common.DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
            device=CUDA_DEVICE,
        ).cpu()

        self.assertTrue(bool(torch.allclose(cpu_repr, gpu_repr, atol=2e-5, rtol=2e-5)))

    def test_gpu_virtual_coil_component_order_matches_cpu(self):
        raw_kspace = torch.view_as_complex(_make_multicoil_kspace(4).contiguous())

        cpu_repr = cpu_common._build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=cpu_common.DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
        )
        gpu_repr = gpu_common._build_kspace_representation(
            raw_kspace,
            target=None,
            uniform_train_resolution=(16, 16),
            representation=gpu_common.DEFAULT_REPRESENTATION,
            num_virtual_coils=4,
            calibration_window=8,
            device=CUDA_DEVICE,
        ).cpu()

        cpu_energies = cpu_repr.square().sum(dim=(1, 2, 3))
        gpu_energies = gpu_repr.square().sum(dim=(1, 2, 3))
        self.assertTrue(bool(torch.allclose(cpu_energies, gpu_energies, atol=1e-5, rtol=1e-5)))


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
class TestGPUShellGram(unittest.TestCase):
    def test_gpu_shell_incremental_gram_matches_cpu_direct_window_gram(self):
        rng = np.random.default_rng(0)
        cache_np = rng.standard_normal((4, 4, 8, 8, 2)).astype(np.float32)

        cumulative_gram = torch.zeros((4, 4), dtype=torch.float64, device=CUDA_DEVICE)
        cumulative_norm_sq = torch.zeros(4, dtype=torch.float64, device=CUDA_DEVICE)
        cache_t = torch.from_numpy(cache_np).to(device=CUDA_DEVICE, dtype=torch.float32)

        for window_size in (2, 4, 6, 8):
            gpu_common._accumulate_shell_gram_inplace(
                cache_t,
                cumulative_gram=cumulative_gram,
                cumulative_norm_sq=cumulative_norm_sq,
                window_size=window_size,
                block_size=2,
                device=CUDA_DEVICE,
            )
            incremental = gpu_common._materialize_window_gram(
                cumulative_gram,
                cumulative_norm_sq,
                normalize_per_sample=True,
            )
            incremental_centered = gpu_common._center_gram_inplace(
                incremental,
                block_size=2,
            ).cpu().numpy()

            direct = cpu_common._compute_window_gram_direct(
                cache_np,
                window_size,
                block_size=2,
                normalize_per_sample=True,
            )
            direct_centered = np.asarray(
                cpu_common._center_gram_inplace(direct.copy(), block_size=2),
                dtype=np.float64,
            )

            np.testing.assert_allclose(
                incremental_centered,
                direct_centered,
                atol=1e-8,
                rtol=1e-6,
            )


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is unavailable")
class TestGPURawCoilGrouping(unittest.TestCase):
    def test_grouped_raw_cache_keeps_native_coil_shapes_without_padding(self):
        dataset = _VariableCoilDataset([3, 5, 3, 5])

        with tempfile.TemporaryDirectory() as tmpdir:
            grouped_caches, sample_groups, cache_mode, cache_paths, _ = (
                gpu_common._load_grouped_raw_coil_kspace_caches(
                    dataset=dataset,
                    num_samples=4,
                    cache_path=Path(tmpdir) / "grouped_cache.dat",
                    uniform_train_resolution=(16, 16),
                    device=CUDA_DEVICE,
                    cache_mode="gpu",
                    gpu_cache_max_gb=60.0,
                )
            )

            self.assertEqual(sample_groups, {3: [0, 2], 5: [1, 3]})
            self.assertEqual(cache_mode, "gpu")
            self.assertEqual(cache_paths, [])
            self.assertEqual(tuple(grouped_caches[3].shape), (2, 3, 16, 16, 2))
            self.assertEqual(tuple(grouped_caches[5].shape), (2, 5, 16, 16, 2))
            self.assertGreater(float(grouped_caches[3][:, -1].abs().sum().item()), 0.0)
            self.assertGreater(float(grouped_caches[5][:, -1].abs().sum().item()), 0.0)

            del grouped_caches
            torch.cuda.empty_cache()

    def test_grouped_raw_hierarchy_writes_group_metadata(self):
        dataset = _VariableCoilDataset([3, 5, 3, 5])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            result = gpu_common.run_hierarchy_job(
                dataset=dataset,
                dataset_tag="variable_coil_grouped_gpu",
                output_prefix=tmpdir_path / "variable_coil_grouped_gpu",
                tmp_dir=tmpdir_path / "scratch",
                num_samples=4,
                uniform_train_resolution=(16, 16),
                normalize_per_sample=True,
                representation=gpu_common.RAW_COIL_REPRESENTATION,
                num_virtual_coils=5,
                calibration_window=8,
                device=CUDA_DEVICE,
                cache_mode="gpu",
                gpu_cache_max_gb=60.0,
                group_raw_coils=True,
                metadata={"source": "raw_kspace"},
            )

            payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertTrue(payload["metadata"]["raw_coil_grouping"])
            self.assertEqual(payload["metadata"]["raw_coil_group_counts"], {"3": 2, "5": 2})
            self.assertEqual(
                payload["metadata"]["raw_coil_group_aggregation"],
                "sample_weighted_block_spectrum",
            )
            self.assertTrue(bool(np.all(np.diff(payload["infos"]) >= -1e-10)))


class _GPUHierarchySmokeMixin:
    def _assert_gpu_hierarchy_job_smoke(self, dataset, dataset_tag: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            result = gpu_common.run_hierarchy_job(
                dataset=dataset,
                dataset_tag=dataset_tag,
                output_prefix=tmpdir_path / f"{dataset_tag}_raw_gpu",
                tmp_dir=tmpdir_path / "scratch",
                num_samples=4,
                uniform_train_resolution=(32, 32),
                normalize_per_sample=True,
                representation=gpu_common.DEFAULT_REPRESENTATION,
                num_virtual_coils=4,
                calibration_window=32,
                device=CUDA_DEVICE,
                cache_mode="auto",
                gpu_cache_max_gb=60.0,
                metadata={
                    "source": "raw_kspace",
                    "representation": gpu_common.DEFAULT_REPRESENTATION,
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
            self.assertEqual(payload["metadata"]["representation"], gpu_common.DEFAULT_REPRESENTATION)
            self.assertEqual(payload["metadata"]["num_virtual_coils"], 4)
            self.assertEqual(payload["metadata"]["calibration_window"], 32)
            self.assertEqual(payload["metadata"]["device"], CUDA_DEVICE)
            self.assertEqual(payload["metadata"]["cache_mode"], "gpu")

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


@unittest.skipUnless(CUDA_AVAILABLE and FASTMRI_ROOT.exists(), "fastMRI or CUDA unavailable")
class TestFastMRIGPUHierarchySmoke(unittest.TestCase, _GPUHierarchySmokeMixin):
    def test_fastmri_multicoil_hierarchy_gpu_smoke(self):
        dataset = SliceDataset(
            root=FASTMRI_ROOT,
            challenge="multicoil",
            transform=None,
        )
        self._assert_gpu_hierarchy_job_smoke(dataset, "fastmri_smoke_gpu")


@unittest.skipUnless(CUDA_AVAILABLE and STANFORD_ROOT.exists(), "Stanford or CUDA unavailable")
class TestStanfordGPUHierarchySmoke(unittest.TestCase, _GPUHierarchySmokeMixin):
    def test_stanford_multicoil_hierarchy_gpu_smoke(self):
        dataset = StanfordSliceDataset(
            root=STANFORD_ROOT,
            data_partition="train",
            train_val_split=0.8,
            train_val_seed=0,
            transform=None,
        )
        self._assert_gpu_hierarchy_job_smoke(dataset, "stanford_smoke_gpu")


class _CPUVsGPUParityMixin:
    def _assert_cpu_gpu_parity(self, dataset, dataset_tag: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cpu_result = cpu_common.run_hierarchy_job(
                dataset=dataset,
                dataset_tag=f"{dataset_tag}_cpu",
                output_prefix=tmpdir_path / f"{dataset_tag}_cpu",
                tmp_dir=tmpdir_path / "cpu_scratch",
                num_samples=4,
                uniform_train_resolution=(32, 32),
                normalize_per_sample=True,
                representation=cpu_common.DEFAULT_REPRESENTATION,
                num_virtual_coils=4,
                calibration_window=32,
                metadata={"parity": "cpu"},
            )
            gpu_result = gpu_common.run_hierarchy_job(
                dataset=dataset,
                dataset_tag=f"{dataset_tag}_gpu",
                output_prefix=tmpdir_path / f"{dataset_tag}_gpu",
                tmp_dir=tmpdir_path / "gpu_scratch",
                num_samples=4,
                uniform_train_resolution=(32, 32),
                normalize_per_sample=True,
                representation=gpu_common.DEFAULT_REPRESENTATION,
                num_virtual_coils=4,
                calibration_window=32,
                device=CUDA_DEVICE,
                cache_mode="auto",
                gpu_cache_max_gb=60.0,
                metadata={"parity": "gpu"},
            )

            cpu_infos = np.asarray(cpu_result["infos"], dtype=np.float64)
            gpu_infos = np.asarray(gpu_result["infos"], dtype=np.float64)
            self.assertLessEqual(float(np.max(np.abs(cpu_infos - gpu_infos))), 1e-5)

            cpu_r_eff = np.asarray(
                [cpu_result["r_eff_by_window"][int(window)] for window in cpu_result["windows"]],
                dtype=np.float64,
            )
            gpu_r_eff = np.asarray(
                [gpu_result["r_eff_by_window"][int(window)] for window in gpu_result["windows"]],
                dtype=np.float64,
            )
            np.testing.assert_allclose(cpu_r_eff, gpu_r_eff, atol=1e-5, rtol=1e-5)

            cpu_schedule = generate_progressive_windows(
                parse_info_file(Path(cpu_result["json_path"])),
                acs_size=16,
                num_blocks=8,
                shell_power=0.25,
                log_alpha=20.0,
                img_size=32,
            )
            gpu_schedule = generate_progressive_windows(
                parse_info_file(Path(gpu_result["json_path"])),
                acs_size=16,
                num_blocks=8,
                shell_power=0.25,
                log_alpha=20.0,
                img_size=32,
            )
            self.assertEqual(cpu_schedule, gpu_schedule)


@unittest.skipUnless(CUDA_AVAILABLE and FASTMRI_ROOT.exists(), "fastMRI or CUDA unavailable")
class TestFastMRICPUVsGPUParity(unittest.TestCase, _CPUVsGPUParityMixin):
    def test_fastmri_cpu_gpu_parity(self):
        dataset = SliceDataset(
            root=FASTMRI_ROOT,
            challenge="multicoil",
            transform=None,
        )
        self._assert_cpu_gpu_parity(dataset, "fastmri_parity")


@unittest.skipUnless(CUDA_AVAILABLE and STANFORD_ROOT.exists(), "Stanford or CUDA unavailable")
class TestStanfordCPUVsGPUParity(unittest.TestCase, _CPUVsGPUParityMixin):
    def test_stanford_cpu_gpu_parity(self):
        dataset = StanfordSliceDataset(
            root=STANFORD_ROOT,
            data_partition="train",
            train_val_split=0.8,
            train_val_seed=0,
            transform=None,
        )
        self._assert_cpu_gpu_parity(dataset, "stanford_parity")


@unittest.skipUnless(
    CUDA_AVAILABLE
    and FASTMRI_ROOT.exists()
    and os.environ.get("PDAC_RUN_GPU_HIERARCHY_BENCHMARK") == "1",
    "Set PDAC_RUN_GPU_HIERARCHY_BENCHMARK=1 to run the optional GPU benchmark helper.",
)
class TestGPUHierarchyBenchmarkHelper(unittest.TestCase):
    def test_fastmri_gpu_benchmark_helper(self):
        dataset = SliceDataset(
            root=FASTMRI_ROOT,
            challenge="multicoil",
            transform=None,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            result = gpu_common.benchmark_hierarchy_job(
                dataset=dataset,
                dataset_tag="fastmri_benchmark_gpu",
                output_prefix=tmpdir_path / "fastmri_benchmark_gpu",
                tmp_dir=tmpdir_path / "scratch",
                num_samples=4,
                uniform_train_resolution=(32, 32),
                normalize_per_sample=True,
                representation=gpu_common.DEFAULT_REPRESENTATION,
                num_virtual_coils=4,
                calibration_window=32,
                device=CUDA_DEVICE,
                cache_mode="auto",
                gpu_cache_max_gb=60.0,
                metadata={"benchmark": True},
            )
            print(json.dumps(result["benchmark"], indent=2, sort_keys=True))
            self.assertGreater(result["benchmark"]["elapsed_seconds"], 0.0)
            self.assertGreaterEqual(result["benchmark"]["peak_cuda_memory_gb"], 0.0)
