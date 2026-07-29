"""One spelling of each temporal resolution, shared by everything that needs it.

`satterc.frequencies` exists because the models, the config generator and the
synthetic data generator all have to agree on what "weekly" means. They used to
carry three separate spellings of ``"7D"`` and ``"1ME"``, kept in step by a
comment. These tests pin the agreement so it cannot drift back apart silently —
the failure mode being a contract mismatch that only shows up at runtime.
"""

import typing

import pytest
from xarray_annotated.temporal import Freq

from satterc import frequencies
from satterc.frequencies import (
    BY_LABEL,
    DAILY,
    MONTHLY,
    WEEKLY,
    offset,
    resample_offset,
)
from satterc.models import pmodel, rothc, sgam, splash


class TestOffsets:
    @pytest.mark.parametrize(
        ("label", "expected"), [("daily", "D"), ("weekly", "7D"), ("monthly", "1ME")]
    )
    def test_offset_of_each_label(self, label, expected):
        assert offset(label) == expected

    def test_unknown_label_names_the_valid_ones(self):
        with pytest.raises(ValueError, match="Unknown frequency label 'yearly'"):
            offset("yearly")

    def test_offsets_are_unanchored(self):
        """A weekly series must be accepted whichever weekday it starts on.

        Pinning the phase (``W-SUN``) would reject a pipeline whose resample
        happens to land on a Wednesday.
        """
        assert offset("weekly") == "7D"
        assert "-" not in offset("weekly")


class TestResampleOffset:
    @pytest.mark.parametrize(
        ("source", "target", "expected"),
        [
            ("daily", "weekly", "7D"),
            ("daily", "monthly", "1ME"),
            ("weekly", "monthly", "1ME"),
        ],
    )
    def test_coarsening_lands_on_the_target(self, source, target, expected):
        assert resample_offset(source, target) == expected

    @pytest.mark.parametrize(
        ("source", "target"),
        [("weekly", "daily"), ("monthly", "weekly"), ("monthly", "daily")],
    )
    def test_refining_is_rejected(self, source, target):
        with pytest.raises(ValueError, match="resampling coarsens"):
            resample_offset(source, target)

    def test_same_label_is_rejected(self):
        with pytest.raises(ValueError, match="resampling coarsens"):
            resample_offset("weekly", "weekly")

    def test_unknown_label_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown frequency label"):
            resample_offset("daily", "yearly")


class TestOneSourceOfTruth:
    """The reason this module exists: three subpackages, one spelling."""

    @pytest.mark.parametrize("module", [pmodel, rothc, sgam, splash])
    def test_models_declare_the_shared_objects(self, module):
        """A model's `Freq` contracts must be these objects, not equal copies.

        Identity, so a model redeclaring `Freq("7D")` locally fails here rather
        than agreeing by coincidence until one of the two is edited.
        """
        declared = {
            value
            for hints in (
                typing.get_type_hints(fn, include_extras=True).values()
                for fn in vars(module).values()
                if callable(fn) and getattr(fn, "__module__", "") == module.__name__
            )
            for hint in hints
            for value in getattr(hint, "__metadata__", ())
            if isinstance(value, Freq)
        }
        assert declared, f"{module.__name__} declares no Freq contracts"
        assert declared <= {DAILY, WEEKLY, MONTHLY}
        assert all(
            any(d is known for known in (DAILY, WEEKLY, MONTHLY)) for d in declared
        )

    @pytest.mark.parametrize("label", ["daily", "weekly", "monthly"])
    def test_the_declared_contract_matches_the_resample_offset(self, label):
        """What data-gen resamples onto is what the models' `Freq` demands."""
        assert offset(label) == str(BY_LABEL[label].freq)

    def test_no_module_redeclares_an_offset(self):
        """Guards against a fourth spelling appearing somewhere.

        Reads the package source rather than any one module, so a new literal
        added anywhere in `satterc` fails here.
        """
        from pathlib import Path

        package = Path(frequencies.__file__).parent
        offenders = [
            path.relative_to(package)
            for path in package.rglob("*.py")
            if path.name != "frequencies.py"
            and any(lit in path.read_text() for lit in ('"7D"', '"1ME"'))
        ]
        assert offenders == [], f"offset literals outside frequencies.py: {offenders}"
