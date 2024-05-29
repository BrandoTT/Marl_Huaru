import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        # self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Sequential(nn.Linear(input_shape, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(256, 1)
                                 )

    def forward(self, batch, t=None):
        inputs = self._build_inputs(batch, t=t)
        q = self.fc1(inputs)
        return q

    def _build_inputs(self, batch, t=None):
        # ts = slice(None) if t is None else slice(t, t+1)
        # return batch["state"][:, ts]
        # new add indepent critic
        ts = slice(None) if t is None else slice(t, t + 1)
        # batch["obs"][:, ts] (2, episode_limit, 5, 129)
        agent_one_hot_id = torch.eye(self.n_agents, device=batch.device)
        agent_one_hot_id = agent_one_hot_id.unsqueeze(0).unsqueeze(0)  # -> (1,1,5,5)
        agent_one_hot_id = agent_one_hot_id.expand(batch["obs"][:, ts].size(0),
                                                   batch["obs"][:, ts].size(1),
                                                   self.n_agents,
                                                   self.n_agents
                                                   )

        critic_inputs_data = torch.cat([batch["obs"], agent_one_hot_id], dim=-1)

        return critic_inputs_data

    def _get_input_shape(self, scheme):
        # input_shape = scheme["state"]["vshape"]
        # return input_shape
        # new add indepent critic
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape