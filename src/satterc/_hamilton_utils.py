"""Utilities for working with Hamilton decorators.

This module provides workarounds for issues with Hamilton's @resolve decorator
when used with extract_fields and parameterize_sources.
"""

from typing import Callable

from hamilton.function_modifiers import extract_fields, parameterize_sources
from hamilton.function_modifiers.base import NodeTransformer
from hamilton.function_modifiers.delayed import resolve, ResolveAt


class NoOpDecorator(NodeTransformer):
    """A no-op decorator that does nothing.

    Used when a decorator needs to be conditionally applied based on config.
    """

    def __init__(self):
        pass

    def validate(self, fn):
        pass

    def transform_node(self, node, config, fn):
        return [node]

    def transform_dag(self, nodes, config, fn):
        return nodes

    def select_nodes(self, target, nodes):
        return []

    def allows_multiple(self):
        return True

    @classmethod
    def get_lifecycle_name(cls):
        return "transform"


class LazyExtractFields(NodeTransformer):
    """A wrapper around extract_fields that defers validation until transform time.

    This is designed to work with the resolve decorator, which doesn't call
    validate() on the returned decorator instance.
    """

    def __init__(self, fields: dict):
        self._fields = fields
        self._ef = None

    def _get_ef(self, fn=None):
        if self._ef is None:
            self._ef = extract_fields(self._fields)
        if fn is not None and not hasattr(self._ef, "resolved_fields"):
            self._ef.validate(fn)
        return self._ef

    def validate(self, fn):
        self._get_ef(fn)

    def transform_node(self, node, config, fn):
        return self._get_ef(fn).transform_node(node, config, fn)

    def transform_dag(self, nodes, config, fn):
        return self._get_ef(fn).transform_dag(nodes, config, fn)

    def select_nodes(self, target, nodes):
        return self._get_ef().select_nodes(target, nodes)

    def allows_multiple(self):
        return self._get_ef().allows_multiple()

    @classmethod
    def get_lifecycle_name(cls):
        return "transform"


def make_extract_fields_resolver(field_key: str, suffix: str = "_daily") -> Callable:
    """Create a resolve decorator for extract_fields based on a config key.

    Parameters
    ----------
    field_key : str
        The config key that contains the list of field names.
    suffix : str
        The suffix to append to field names (e.g., "_daily").

    Returns
    -------
    Callable
        A decorator that can be used with @resolve.

    Example
    -------
    @make_extract_fields_resolver("daily", "_daily")
    def unpack_daily_inputs(daily_inputs_stacked, daily):
        ...
    """

    def decorator_factory(daily):
        return LazyExtractFields({f"{var}{suffix}": _get_field_type() for var in daily})

    return resolve(
        when=ResolveAt.CONFIG_AVAILABLE,
        decorate_with=decorator_factory,
    )


def _get_field_type():
    """Placeholder for field type. Returns xr.DataArray by default."""
    import xarray as xr

    return xr.DataArray


def make_parameterize_sources_resolver(
    config_key: str, source_suffix: str = "_daily", target_suffix: str = "_daily"
) -> Callable:
    """Create a resolve decorator for parameterize_sources based on a config key.

    Parameters
    ----------
    config_key : str
        The config key that contains the list of variable names.
    source_suffix : str
        The suffix on source variable names.
    target_suffix : str
        The suffix on target variable names.

    Returns
    -------
    Callable
        A decorator that can be used with @resolve.

    Example
    -------
    @make_parameterize_sources_resolver("weekly_from_daily", "_daily", "_weekly")
    def aggregate_daily_to_weekly(var_daily):
        ...
    """

    def decorator_factory(vars_list):
        return parameterize_sources(
            **{
                f"{var}{target_suffix}": {"var_daily": f"{var}{source_suffix}"}
                for var in vars_list
            }
        )

    return resolve(
        when=ResolveAt.CONFIG_AVAILABLE,
        decorate_with=decorator_factory,
    )
