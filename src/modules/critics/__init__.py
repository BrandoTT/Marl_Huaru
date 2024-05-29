from .centralv import CentralVCritic
from .coma import COMACritic
from .fmac_critic import FMACCritic
from .lica import LICACritic
from .offpg import OffPGCritic

REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["fmac_critic"] = FMACCritic
REGISTRY["lica_critic"] = LICACritic
REGISTRY["offpg_critic"] = OffPGCritic
