
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class NRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(NRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        # 多分类头设计
        self.head_1 = nn.Linear(args.rnn_hidden_dim, args.n_actions_1)
        self.head_2 = nn.Linear(args.rnn_hidden_dim, args.n_actions_2)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)
        
        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.head_1, gain=args.gain)
            orthogonal_init_(self.head_2, gain=args.gain)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, e = inputs.size()
        inputs = inputs.view(-1, e)
        # normalize -> [-1, 1]

        x = F.relu(self.fc1(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, h_in)

        if getattr(self.args, "use_layer_norm", False):
            q_1 = self.head_1(self.layer_norm(hh))
            q_2 = self.head_2(self.layer_norm(hh))
        else:
            q_1 = self.head_1(hh)
            q_2 = self.head_2(hh)

        # 如果是Actor-Critic的算法，那么q就是动作输出概率
        q_1 = q_1.view(b, a, -1)
        q_2 = q_2.view(b, a, -1)

        return q_1, q_2, hh.view(b, a, -1)
