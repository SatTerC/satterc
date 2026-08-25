"""Regenerate ``tests/data/pyrealm_golden.json``, the pyrealm unit anchor.

The golden file pins *absolute* pyrealm output magnitudes for a fixed set of
inputs, alongside the unit strings satterc declares for those outputs.
`tests/test_pyrealm_units.py` checks both, so a number and its label cannot
drift apart unnoticed.

Run this ONLY when a pyrealm change has been reviewed and the new values are
understood. Regenerating to turn a red test green destroys the signal the file
exists to carry. If pyrealm changed a unit convention, correct satterc's
annotation rather than re-recording the numbers.

    just regen-pyrealm-golden
"""

import json
import sys
from pathlib import Path
from typing import Any

import pyrealm

REPO_ROOT = Path(__file__).parent.parent
GOLDEN_PATH = REPO_ROOT / "tests" / "data" / "pyrealm_golden.json"

# The cases live with the test that consumes them, so the two cannot diverge.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from pyrealm_cases import (  # noqa: E402
    declared_units,
    pmodel_inputs,
    splash_inputs,
    summarize,
)

from satterc.models.pmodel import PModelOut, _pmodel  # noqa: E402
from satterc.models.splash import SplashOut, _splash  # noqa: E402


def build() -> dict[str, Any]:
    pmodel_result = _pmodel(**pmodel_inputs())
    splash_result = _splash(**splash_inputs())
    return {
        "pyrealm_version": pyrealm.__version__,
        "_comment": (
            "Regenerate with `just regen-pyrealm-golden`, and only after "
            "reviewing why pyrealm's output changed. If a unit convention "
            "moved, correct satterc's annotation rather than these numbers."
        ),
        "pmodel": {
            "units": declared_units(PModelOut),
            "outputs": {name: summarize(da) for name, da in pmodel_result.items()},
        },
        "splash": {
            "units": declared_units(SplashOut),
            "outputs": {name: summarize(da) for name, da in splash_result.items()},
        },
    }


def main() -> None:
    GOLDEN_PATH.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"Wrote {GOLDEN_PATH} for pyrealm {pyrealm.__version__}")


if __name__ == "__main__":
    main()
