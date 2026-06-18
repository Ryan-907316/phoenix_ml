# cli.py
# Console-script entry point: called when the user runs `phoenix-ml` after pip install.
# Contains the same splash + launch logic as app.py so that both routes work identically.

import matplotlib
matplotlib.use('Agg')  # must be set before any other matplotlib/phoenix_ml imports

import sys
import warnings
warnings.filterwarnings("ignore")

from phoenix_ml.system_info import SystemInfo
from phoenix_ml.ui import launch

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
