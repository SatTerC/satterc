"""Unit bridges between what a model reports and what its consumer wants.

Each model wrapper reports the units its upstream library actually uses rather
than rescaling, so a producer and a consumer of the same quantity can disagree.
This table lists those pairs and the factor between them: the config generator
reads it to emit a ``[[node]]`` doing the conversion, and `recipes/config.toml`
writes the same nodes by hand.

A bridge is only consulted for an input no model produces under its own name. It
does not coarsen — `source` and `target` must be wanted at the same frequency —
so it is a pure restatement of units.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Bridge:
    """One producer/consumer pair whose declared units disagree.

    Attributes
    ----------
    source : str
        Base name (no frequency suffix) of the node that is produced.
    target : str
        Base name of the input a model wants and nothing produces.
    factor : float | None
        Multiplier taking `source` units to `target` units. ``None`` where the
        two are dimensionally the same quantity, in which case the generator
        asks pint for the factor rather than repeating a number that a unit
        registry already knows.
    note : str
        Why the factor is what it is, for the reader of this table. Mirrors the
        prose in `recipes/config.toml`.
    """

    source: str
    target: str
    factor: float | None = None
    note: str = ""


#: Every bridge satterc knows about. Adding a model whose units disagree with a
#: consumer's means adding a row here, not editing the generator.
BRIDGES: tuple[Bridge, ...] = (
    Bridge(
        source="gpp_flux",
        target="gpp",
        note=(
            "pyrealm reports GPP as an instantaneous flux in ug C m-2 s-1; SGAM "
            "wants a daily rate in g C m-2 d-1. Dimensionally the same quantity, "
            "so pint supplies the 86400 s d-1 x 1e-6 g ug-1 = 0.0864."
        ),
    ),
    Bridge(
        source="lue_photon",
        target="lue",
        factor=4.57,
        note=(
            "pyrealm defines LUE against PPFD, a *photon* flux, so its "
            "denominator counts moles of photons; SGAM wants carbon per MJ of "
            "absorbed PAR. Not a unit conversion — the two are bridged by the "
            "photon content of PAR, ~4.57 mol MJ-1 over the 400-700 nm band."
        ),
    ),
)


def bridge_for(target: str) -> Bridge | None:
    """Return the bridge producing ``target``, or None if nothing does."""
    for bridge in BRIDGES:
        if bridge.target == target:
            return bridge
    return None
