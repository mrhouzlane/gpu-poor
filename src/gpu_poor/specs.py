"""Vendor peak rates. SI GB/TB, dense BF16 Tensor rates, no sparsity."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    memory_gb: float
    bandwidth_gb_s: float
    bf16_tflops: float
    source: str

    @property
    def memory_bytes(self) -> int:
        return int(self.memory_gb * 10**9)

    @property
    def bandwidth_bytes_s(self) -> float:
        return self.bandwidth_gb_s * 10**9


# H100/H200 datasheets advertise sparse BF16 1,979 TFLOPS: divide by 2.
# https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306
# https://www.nvidia.com/en-us/data-center/h200/
# A100 80GB SXM (NOT PCIe): 2,039 GB/s, dense BF16 312 TFLOPS.
# https://resources.nvidia.com/en-us-gpu/nvidia-a100-datashee-1
# L40S page explicitly lists dense BF16 362.05 (and sparse 733).
# https://www.nvidia.com/en-us/data-center/l40s/
# RTX 4090: Ada whitepaper Appendix A, BF16 with FP32 accumulation,
# dense 165.2 TFLOPS; 24 GB; 1,008 GB/s. NOT the INT8 marketing TOPS.
# https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
# MI300X datasheet: dense BF16 1,307.4 TFLOPS, 192 GB, 5.3 TB/s.
# https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
GPUS: dict[str, GPU] = {
    "H100": GPU(
        "H100 SXM", 80, 3350, 989.5, "https://resources.nvidia.com/en-us-gpu/h100-datasheet-24306"
    ),
    "H200": GPU("H200 SXM", 141, 4800, 989.5, "https://www.nvidia.com/en-us/data-center/h200/"),
    "A100": GPU(
        "A100 80GB SXM",
        80,
        2039,
        312,
        "https://resources.nvidia.com/en-us-gpu/nvidia-a100-datashee-1",
    ),
    "L40S": GPU("L40S", 48, 864, 362.05, "https://www.nvidia.com/en-us/data-center/l40s/"),
    "RTX4090": GPU(
        "RTX 4090",
        24,
        1008,
        165.2,
        "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
    ),
    "MI300X": GPU(
        "MI300X",
        192,
        5300,
        1307.4,
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/"
        "data-sheets/amd-instinct-mi300x-data-sheet.pdf",
    ),
}


def get_gpu(name: str) -> GPU:
    key = name.upper().replace(" ", "").replace("_", "").replace("-", "")
    aliases = {
        "H100SXM": "H100",
        "H200SXM": "H200",
        "A10080GB": "A100",
        "A10080GBSXM": "A100",
        "H100SXM5": "H100",
    }
    key = aliases.get(key, key)
    if key not in GPUS:
        raise ValueError(f"Unknown GPU {name!r}; choose {', '.join(GPUS)}")
    return GPUS[key]
