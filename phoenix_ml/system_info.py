# system_info.py
# This module collects some basic information about the device you are using to give the user more information about what they can run.
# All this information is displayed in the report as a neat table.

import os
import sys
import platform
import cpuinfo
import psutil
import win32com.client
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


class SystemInfo:
    # Collects and displays system information (OS, CPU, RAM, GPU, Python version).

    def __init__(self):
        self.info = {
            "Feature": [],
            "Details": []
        }

    def gather(self):
        self.info["Feature"] = [
            "Operating System",
            "Processor",
            "CPU",
            "Number of Physical Cores",
            "Threads",
            "Python Version",
            "RAM"
        ]

        self.info["Details"] = [
            f"{platform.system()} {platform.release()}",
            platform.processor(),
            cpuinfo.get_cpu_info().get("brand_raw", "Unknown"),
            psutil.cpu_count(logical=False),
            psutil.cpu_count(logical=True),
            sys.version.split()[0],
            f"{round(psutil.virtual_memory().total / 1e9, 2)} GB" # Note: Total available RAM, not the advertised amount
        ]

        # GPU Detection using WMI (Windows only)
        try:
            wmi = win32com.client.GetObject("winmgmts:")
            gpu_list = [gpu.Name.strip() for gpu in wmi.InstancesOf("Win32_VideoController")]
            gpu_details = ", ".join(gpu_list) if gpu_list else "No dedicated GPU detected" # If user isn't on Windows then it won't work :(
        except Exception as e:
            gpu_details = f"GPU detection failed: {e}"

        self.info["Feature"].append("GPU")
        self.info["Details"].append(gpu_details)

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
