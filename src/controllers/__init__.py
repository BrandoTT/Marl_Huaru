REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .lica_controller import LICAMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["lica_mac"] = LICAMAC