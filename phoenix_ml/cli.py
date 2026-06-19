# cli.py
# Console-script entry point: called when the user runs `phoenix-ml` after pip install.

import sys
import argparse
from pathlib import Path


def _get_examples():
    import shutil
    src = Path(__file__).parent / "examples"
    dst = Path.cwd() / "examples"
    if dst.exists():
        print(f"  'examples' folder already exists at:\n  {dst}")
        print("  Delete or rename it first if you want a fresh copy.")
    else:
        shutil.copytree(src, dst)
        print(f"  Examples copied to:\n  {dst}")
        print()
        print("  Datasets are in:  examples/Original Datasets/")
        print("  To generate the DC motor dataset manually, run:")
        print("    python examples/DC_Motors_Dataset_Generation.py")
        print()
        print("  Open phoenix-ml and point the Dataset Path at one of those CSV files.")


_BOLD_ON  = "\x1b[1m"
_BOLD_OFF = "\x1b[0m"

def _bold_letters(text: str, letters: str, case_insensitive: bool = True) -> str:
    target = set(letters.lower() if case_insensitive else letters)
    out = []
    for ch in text:
        if (ch.lower() if case_insensitive else ch) in target:
            out.append(f"{_BOLD_ON}{ch}{_BOLD_OFF}")
        else:
            out.append(ch)
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="phoenix-ml",
        description="Phoenix ML — Physics-Enhanced Machine Learning workflow.",
        add_help=True,
    )
    parser.add_argument(
        "--get-examples",
        action="store_true",
        help="Copy the bundled example datasets to the current directory and exit.",
    )
    args, _ = parser.parse_known_args()

    if args.get_examples:
        _get_examples()
        return

    import matplotlib
    matplotlib.use('Agg')
    import warnings
    warnings.filterwarnings("ignore")

    from phoenix_ml.system_info import SystemInfo
    from phoenix_ml.ui import launch

    si = SystemInfo()
    si.gather()

    print()
    print("=" * 70)
    desc = ("  phoenix_ml  --  A Physics and Hybrid Optimised ENgine for "
            "Interpretability and eXplainability for Machine Learning")
    print(_bold_letters(desc, "phoenix_ml"))
    print("=" * 70)
    print()
    si.display()
    print("=" * 70)

    print("\n  Press any key to open the interface...")
    if sys.platform == "win32":
        import msvcrt
        msvcrt.getch()
    else:
        input()

    launch()
