"""Tests for satterc.dag._hamilton_fixes."""

import pytest

from satterc.dag._hamilton_fixes import NoOpDecorator


class TestNoOpDecorator:
    """NoOpDecorator is a transparent Hamilton NodeTransformer."""

    @pytest.fixture
    def dec(self):
        return NoOpDecorator()

    def test_validate_does_not_raise(self, dec):
        dec.validate(lambda: None)

    def test_transform_node_wraps_in_list(self, dec):
        sentinel = object()
        result = dec.transform_node(sentinel, {}, lambda: None)
        assert result == [sentinel]

    def test_transform_dag_returns_same_collection(self, dec):
        nodes = [object(), object()]
        result = dec.transform_dag(nodes, {}, lambda: None)
        assert result is nodes

    def test_select_nodes_is_empty(self, dec):
        nodes = [object(), object()]
        result = NoOpDecorator.select_nodes(None, nodes)
        assert list(result) == []

    def test_allows_multiple_true(self):
        assert NoOpDecorator.allows_multiple() is True

    def test_lifecycle_name_is_transform(self):
        assert NoOpDecorator.get_lifecycle_name() == "transform"
