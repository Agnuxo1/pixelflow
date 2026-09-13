"""pixelflow.core: reservoir abstraction, CA rules, and input encoders."""

from pixelflow.core.reservoir import Reservoir, ReservoirConfig
from pixelflow.core.rules import RuleSpec, get_rule, list_rules, register_rule

__all__ = [
    "Reservoir",
    "ReservoirConfig",
    "RuleSpec",
    "get_rule",
    "list_rules",
    "register_rule",
]
