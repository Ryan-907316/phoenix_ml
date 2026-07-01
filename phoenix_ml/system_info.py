# system_info.py
# Collects basic system information and displays it at startup and in the PDF report.

import os
import sys
import platform
import cpuinfo
import psutil
try:
    import win32com.client as _win32com
except ImportError:
    _win32com = None
import warnings
import pandas as pd


class SuppressLibraryLogs:
    # Context manager to suppress stdout and stderr from noisy libraries.
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def _detect_gpu():
    if _win32com is not None:
        try:
            wmi = _win32com.GetObject("winmgmts:")
            gpu_list = [gpu.Name.strip() for gpu in wmi.InstancesOf("Win32_VideoController")]
            return ", ".join(gpu_list) if gpu_list else "No dedicated GPU detected"
        except Exception as e:
            return f"GPU detection failed: {e}"

    if sys.platform.startswith("linux"):
        try:
            import subprocess
            result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
            gpus = [
                line.split(": ", 1)[-1].strip()
                for line in result.stdout.splitlines()
                if any(k in line for k in ("VGA", "3D", "Display"))
            ]
            return ", ".join(gpus) if gpus else "No dedicated GPU detected"
        except Exception:
            return "GPU detection not available"

    if sys.platform == "darwin":
        try:
            import subprocess
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10,
            )
            gpus = [
                line.split(":", 1)[1].strip()
                for line in result.stdout.splitlines()
                if "Chipset Model:" in line
            ]
            return ", ".join(gpus) if gpus else "No dedicated GPU detected"
        except Exception:
            return "GPU detection not available"

    return "GPU detection not available"


class SystemInfo:
    # Collects and displays system information (OS, CPU, RAM, GPU, Python version).

    def __init__(self):
        self.info = {
            "Feature": [],
            "Details": []
        }

    def gather(self):
        freq = psutil.cpu_freq()
        cpu_freq = f"{freq.current / 1000:.2f} GHz" if freq else "Unknown"

        mem = psutil.virtual_memory()
        total_ram = f"{round(mem.total / (1024**3), 1)} GiB"
        available_ram = f"{round(mem.available / (1024**3), 1)} GiB"

        disk = psutil.disk_usage(os.path.expanduser("~"))
        free_disk = (
            f"{round(disk.free / (1024**3), 1)} GiB free "
            f"/ {round(disk.total / (1024**3), 1)} GiB total"
        )

        self.info["Feature"] = [
            "Operating System",
            "Processor",
            "CPU",
            "Physical Cores",
            "Threads",
            "CPU Frequency",
            "Python Version",
            "RAM",
            "RAM Available",
            "Free Disk Space",
            "GPU",
        ]

        self.info["Details"] = [
            f"{platform.system()} {platform.release()}",
            platform.processor(),
            cpuinfo.get_cpu_info().get("brand_raw", "Unknown"),
            psutil.cpu_count(logical=False),
            psutil.cpu_count(logical=True),
            cpu_freq,
            sys.version.split()[0],
            total_ram,
            available_ram,
            free_disk,
            _detect_gpu(),
        ]

        return self.info

    def display(self):
        print("System Information:\n")
        df = pd.DataFrame(self.info)
        print(df.to_string(index=False))

        print(
            "\nNote: GPU acceleration in this workflow is optimised for NVIDIA GPUs using CUDA.\n"
            "This is because popular Machine Learning frameworks like PyTorch rely on CUDA, a proprietary technology developed by NVIDIA for GPU acceleration.\n"
            "While there are alternative frameworks and libraries, such as ROCm for AMD GPUs or oneAPI for Intel GPUs, these are not yet universally supported or integrated in many ML workflows.\n"
            "As a result, this workflow defaults to CUDA for GPU acceleration.\n\n"
            "For systems with AMD GPUs, users may explore ROCm for compatibility with specific frameworks. Similarly, Intel GPU users can consider Intel oneAPI. Note that additional setup may be required to enable GPU support with these alternatives.\n"
            "If no compatible GPU is detected, the workflow will default to using the CPU, which may significantly increase computation time.\n\n"
            "For more details on GPU support, you can explore the following resources:\n"
            "CUDA (NVIDIA): https://docs.nvidia.com/cuda/ \n"
            "ROCm (AMD): https://rocm.docs.amd.com/en/latest/ \n"
            "oneAPI (Intel): https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html \n"
        )


# Local test
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    system_info = SystemInfo()
    system_info.gather()
    system_info.display()
