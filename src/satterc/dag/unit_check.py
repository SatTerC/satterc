"""Build-time (static) unit-consistency check for the Hamilton DAG.

The runtime check (``declare_units`` → ``check_units``) only fires when a node
executes. :func:`check_dag_units` adds the complementary guarantee at *build*
time: every internal edge whose producer and consumer both declare a unit is
verified for consistency, so a mismatch surfaces as soon as the driver is built
rather than part way through a run.

Declarations are read from each node's public function signature via
:func:`satterc.units.units_from_signature` — the same single source the runtime
check uses. Hamilton unifies nodes by name, so a node name that is both
*produced* with a declared unit (a ``TypedDict`` field or bare ``Annotated``
return) and *consumed* with a declared unit (an ``Annotated`` parameter) is a
genuine edge in the built graph; no edge walking is required.

**Limitation.** Edges routed through resample/derive nodes, or fed by external
files, have no statically declared producer unit (resample preserves the source
``units`` attribute at runtime; derive modules are generated; file inputs are
validated against the file's own ``units``). Those edges are not checked here and
fall back to the runtime check.
"""

import warnings
from typing import TYPE_CHECKING

from hamilton import graph_types

from ..units import (
    Mode,
    get_exact_match,
    get_mode,
    units_compatible,
    units_equal,
    units_from_signature,
)

if TYPE_CHECKING:
    from hamilton import driver


def check_dag_units(
    dr: "driver.Driver",
    *,
    mode: Mode | None = None,
    exact: bool | None = None,
) -> None:
    """Verify declared units are consistent across the built DAG's edges.

    For every node name that is both produced and consumed with a declared unit
    (and for external inputs shared by multiple consumers), the declared units
    are compared:

    - dimensionally **incompatible** units (e.g. a mass where a pressure is
      declared) are always reported;
    - dimensionally compatible but **non-identical** strings (e.g. ``"Pa"`` vs
      ``"hPa"``) are reported only when ``exact`` is enabled.

    Behaviour follows the active validation mode (resolved from
    :func:`satterc.units.get_mode` when ``mode`` is ``None``): ``off`` skips the
    check entirely, ``warn`` emits a warning listing the findings, ``strict``
    raises :class:`ValueError`. ``exact`` defaults to
    :func:`satterc.units.get_exact_match`.
    """
    mode = mode or get_mode()
    if mode == "off":
        return
    exact = get_exact_match() if exact is None else exact

    hg = graph_types.HamiltonGraph.from_graph(dr.graph)

    # Distinct public node functions present in this pipeline (dedup by identity).
    seen: set[int] = set()
    funcs = []
    for node in hg.nodes:
        for fn in node.originating_functions or ():
            if id(fn) not in seen:
                seen.add(id(fn))
                funcs.append(fn)

    # name -> (declared_unit, producer_label)
    produced: dict[str, tuple[str, str]] = {}
    # name -> [(declared_unit, consumer_label), ...]
    consumed: dict[str, list[tuple[str, str]]] = {}

    for fn in funcs:
        fn_name = getattr(fn, "__name__", repr(fn))
        in_units, out_units = units_from_signature(fn)
        if isinstance(out_units, dict):
            for name, unit in out_units.items():
                produced[name] = (unit, fn_name)
        elif isinstance(out_units, str):
            # Single-output node: the node name equals the function name.
            produced[fn_name] = (out_units, fn_name)
        for name, unit in in_units.items():
            consumed.setdefault(name, []).append((unit, fn_name))

    # Resampling is unit-preserving (the node copies the source's `units` attr),
    # so a resample target's unit equals its source's. Propagate to a fixpoint so
    # resampled edges become checkable; chained resamples resolve over iterations.
    resample_edges = {
        node.name: next(iter(node.required_dependencies))
        for node in hg.nodes
        if node.tags.get("module") == "satterc.dag.resample"
        and len(node.required_dependencies) == 1
    }
    changed = True
    while changed:
        changed = False
        for target, source in resample_edges.items():
            if target not in produced and source in produced:
                produced[target] = (produced[source][0], f"resample of {source}")
                changed = True

    findings: list[str] = []
    for name, consumers in consumed.items():
        candidates: list[tuple[str, str]] = []
        if name in produced:
            unit, who = produced[name]
            candidates.append((unit, f"output of {who}"))
        candidates.extend((unit, f"input of {who}") for unit, who in consumers)
        if len(candidates) < 2:
            continue  # external input / single consumer — nothing to compare

        base_unit, base_src = candidates[0]
        for unit, src in candidates[1:]:
            if not units_compatible(base_unit, unit):
                findings.append(
                    f"  {name!r}: {base_src} declares {base_unit!r} but {src} "
                    f"declares {unit!r} (dimensionally incompatible)"
                )
            elif exact and not units_equal(base_unit, unit):
                findings.append(
                    f"  {name!r}: {base_src} declares {base_unit!r} but {src} "
                    f"declares {unit!r} (units differ; exact match required)"
                )

    if not findings:
        return
    message = "unit declaration mismatch(es) in DAG:\n" + "\n".join(findings)
    if mode == "strict":
        raise ValueError(message)
    warnings.warn(message, stacklevel=2)
