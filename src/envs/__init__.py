from functools import partial

from .multiagentenv import MultiAgentEnv
from .battle5v5.env.environment import HuaRuBattleEnvWrapper

def env_fn(env, logger=None, address=None, **kwargs) -> MultiAgentEnv:
    print(f'Address is : {address}')
    return env(logger=logger, address=address, **kwargs)

REGISTRY = {}
# REGISTRY["hok"] = partial(env_fn, env=HokEnv)
REGISTRY["huarubattle"] = partial(env_fn, env=HuaRuBattleEnvWrapper)