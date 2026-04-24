"""pixelflow.core: reservoir abstraction, CA rules, and input encoders."""

from pixelflow.core.reservoir import Reservoir, ReservoirConfig
from pixelflow.core.rules import get_rule, list_rules, register_rule, RuleSpec

__all__ = [
    "Reservoir",
    "ReservoirConfig",
    "get_rule",
    "list_rules",
    "register_rule",
    "RuleSpec",
]
