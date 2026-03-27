from typing import Any

from hamilton.data_quality.base import DataValidator, ValidationResult
import pandas as pd
import xarray as xr


class DatetimeIndexValidator(DataValidator):
    def __init__(self, freq: str) -> None:
        super().__init__(importance="fail")

        if freq not in ("D", "W-SUN", "ME"):
            raise ValueError("`freq` must be one of 'D', 'W-SUN', or 'ME'")

        self.freq = freq

    def applies_to(self, datatype) -> bool:
        return issubclass(datatype, pd.Index)

    def description(self) -> str:
        return "Ensures the supplied index is a pandas DatetimeIndex with the expected frequency."

    @classmethod
    def name(cls) -> str:
        return "datetimeindex_validator"

    def validate(self, dataset: Any) -> ValidationResult:
        index = dataset

        if not isinstance(index, pd.DatetimeIndex):
            return ValidationResult(
                passes=False, message=f"Expected pd.DatetimeIndex, got {type(index)}"
            )

        if not (freq := (index.freqstr or pd.infer_freq(index))) == self.freq:
            return ValidationResult(
                passes=False, message=f"Expected frequency {self.freq}, got {freq}"
            )

        return ValidationResult(passes=True, message="")
