
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np

# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        # avail_actions = ep_batch["avail_actions"][:, t_ep]
        avail_actions_1 = ep_batch["avail_actions_1"][:,t_ep]
        avail_actions_2 = ep_batch["avail_actions_2"][:,t_ep]

        agent_outputs_1, agent_outputs_2 = self.forward(ep_batch, t_ep, test_mode=test_mode)

        chosen_actions_1 = self.action_selector.select_action(agent_outputs_1[bs], avail_actions_1[bs], t_env, test_mode=test_mode)
        chosen_actions_2 = self.action_selector.select_action(agent_outputs_2[bs], avail_actions_2[bs], t_env, test_mode=test_mode)

        return chosen_actions_1, chosen_actions_2

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        # # ---------------- Normalize -------------------
        # agent_inputs_mean = agent_inputs.mean(axis=(0, 1), keepdim=True)
        # agent_inputs_std = agent_inputs.mean(axis=(0, 1), keepdim=True)
        # # 避免标准差为零的情况
        # agent_inputs_std[agent_inputs_std == 0] = 1
        # # 对数据进行归一化
        # agent_inputs = (agent_inputs - agent_inputs_mean) / agent_inputs_std

        # # 计算每个特征的最小值和最大值
        # agent_inputs_min = agent_inputs.min(dim=0, keepdim=True)[0]
        # agent_inputs_min = agent_inputs_min.min(dim=1, keepdim=True)[0]
        #
        # agent_inputs_max = agent_inputs.max(dim=0, keepdim=True)[0]
        # agent_inputs_max = agent_inputs_max.max(dim=1, keepdim=True)[0]
        #
        # # 避免最大值等于最小值的情况，防止除以零
        # agent_inputs_range = agent_inputs_max - agent_inputs_min
        # agent_inputs_range[agent_inputs_range == 0] = 1
        # # 将数据缩放到 [0, 1] 区间
        # agent_inputs = (agent_inputs - agent_inputs_min) / agent_inputs_range
        # # 将数据缩放到 [-1, 1] 区间
        # agent_inputs = agent_inputs * 2 - 1

        avail_actions_1 = ep_batch["avail_actions_1"][:, t]
        avail_actions_2 = ep_batch["avail_actions_2"][:, t]

        if test_mode:
            self.agent.eval()

        # agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        agent_outs_1, agent_outs_2, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs_1 = agent_outs_1.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions_1 = avail_actions_1.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs_1[reshaped_avail_actions_1 == 0] = -1e5

                agent_outs_2 = agent_outs_2.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions_2 = avail_actions_2.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs_2[reshaped_avail_actions_2 == 0] = -1e5

            agent_outs_1 = th.nn.functional.softmax(agent_outs_1, dim=-1)
            agent_outs_2 = th.nn.functional.softmax(agent_outs_2, dim=-1)
            
        return agent_outs_1.view(ep_batch.batch_size, self.n_agents, -1), agent_outs_2.view(ep_batch.batch_size, self.n_agents, -1),

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        print('已经成功载入模型！！！')
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)
        # print(f'actor obs inputs size: {inputs.shape}')
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
