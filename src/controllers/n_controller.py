
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class NMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(NMAC, self).__init__(scheme, groups, args)
        
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # avail_actions = ep_batch["avail_actions"][:, t_ep]
        avail_actions_1 = ep_batch["avail_actions_1"][:, t_ep]
        avail_actions_2 = ep_batch["avail_actions_2"][:, t_ep]

        qvals_1, qvals_2 = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # chosen_actions = self.action_selector.select_action(qvals[bs], avail_actions[bs], t_env, test_mode=test_mode)
        chosen_actions_1 = self.action_selector.select_action(qvals_1[bs], avail_actions_1[bs], t_env, test_mode=test_mode)
        chosen_actions_2 = self.action_selector.select_action(qvals_2[bs], avail_actions_2[bs], t_env, test_mode=test_mode)

        return chosen_actions_1, chosen_actions_2

    def forward(self, ep_batch, t, test_mode=False):
        if test_mode:
            self.agent.eval()
        agent_inputs = self._build_inputs(ep_batch, t)
        # ---------------- Normalize -------------------
        agent_inputs_mean = agent_inputs.mean(axis=(0, 1), keepdim=True)
        agent_inputs_std = agent_inputs.mean(axis=(0, 1), keepdim=True)
        # 避免标准差为零的情况
        agent_inputs_std[agent_inputs_std == 0] = 1
        # 对数据进行归一化
        agent_inputs = (agent_inputs - agent_inputs_mean) / agent_inputs_std

        # avail_actions = ep_batch["avail_actions"][:, t]
        avail_actions_1 = ep_batch["avail_actions_1"][:, t]
        avail_actions_2 = ep_batch["avail_actions_2"][:, t]

        # agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs_1, agent_outs_2, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        return agent_outs_1, agent_outs_2
