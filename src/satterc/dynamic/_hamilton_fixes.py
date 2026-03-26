from hamilton.function_modifiers.base import NodeTransformer
from hamilton.function_modifiers.delayed import resolve


class FixedResolve(resolve):
    """A fix for @resolve that properly validates the returned decorator.

    Hamilton's @resolve decorator doesn't call validate() on decorators
    returned from decorate_with. This subclass fixes that.

    Should be fixed upstream at some point - see https://github.com/apache/hamilton/pull/1524
    """

    def resolve(self, config, fn):
        decorator = super().resolve(config, fn)
        if hasattr(decorator, "validate"):
            decorator.validate(fn)
        return decorator


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
