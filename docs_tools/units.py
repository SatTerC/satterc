"""Griffe extension that turns a `TypedDict` of outputs into a table.

SatTerC model nodes return a `TypedDict` whose fields carry their type, units
and frequency in the annotation, as ``Annotated[DataArray, "mm d-1", DAILY]``,
and their meaning in an attribute docstring. Rendered as class members that
becomes a run of headings; what the model pages want is one table, matching the
one the inputs get from the node signature.

This extension rewrites the class docstring to carry an ``Attributes`` section
built from those fields, each keeping its annotation, so the docs render the
outputs through the same machinery as the inputs.
"""

from __future__ import annotations

from textwrap import indent
from typing import Any

import griffe

logger = griffe.get_logger(__name__)


def _is_typed_dict(cls: griffe.Class) -> bool:
    return any(str(base).rsplit(".", 1)[-1] == "TypedDict" for base in cls.bases)


class OutputsTableExtension(griffe.Extension):
    """Rewrite `TypedDict` fields as an ``Attributes`` docstring section."""

    def on_class_members(self, *, cls: griffe.Class, **kwargs: Any) -> None:
        """Append an ``Attributes`` section listing the fields and their annotations."""
        if not _is_typed_dict(cls) or cls.docstring is None:
            return
        if "Attributes\n" in cls.docstring.value:
            return

        rows = []
        for member in cls.members.values():
            if not isinstance(member, griffe.Attribute) or member.docstring is None:
                continue
            rows.append(
                f"{member.name} : {member.annotation}\n"
                + indent(member.docstring.value, "    ")
            )

        if not rows:
            return

        cls.docstring.value = (
            cls.docstring.value.rstrip()
            + "\n\nAttributes\n----------\n"
            + "\n".join(rows)
            + "\n"
        )
