import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralv import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from components.standarize_stream import RunningMeanStd


class PPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        # self.n_actions = args.n_actions

        self.n_actions_1 = args.n_actions_1
        self.n_actions_2 = args.n_actions_2

        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        import numpy as np
        rewards = batch["reward"][:, :-1]  # [4, 898, 1]
        # 每个Head对应一个actions
        # actions = batch["actions"][:, :]
        actions_1 = batch["actions_1"][:, :-1]  # ([4, 899, 5, 1]) 和reward是统一长度
        actions_2 = batch["actions_2"][:, :-1]   #
        terminated = batch["terminated"][:, :-1].float()  # (4, 1199, 1) terminated都是按照max_seq_len创建的，
        mask = batch["filled"][:, :-1].float()  # what is that filled mean???? (4, 1199, 1)
        # 猜测这个batch["filled"]可能是mask掩码，就是一个batch中，某些样本的长度不够max_seq_length，所以给mask掉
        # mask可以理解成是有效路径，就是说terminated是到了最后一步是1，
        # mask 是指从0到他的最后一步都是1
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # 这一步的目的是什么？
        # 1 - terminated[:, :-1] 会造成短轨迹样本，中间的截至点（1）为0,然后两边都是1
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) 这个目的是让所有的真实轨迹点都为1

        # actions = actions[:, :-1]
        # actions_1 = actions_1[:, :-1]
        # actions_2 = actions_2[:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # No experiences to train on in this minibatch
        if mask.sum() == 0:  # 是说一步都没有走，全是0
            self.logger.log_stat("Mask_Sum_Zero", 1, t_env)
            self.logger.console_logger.error("Actor Critic Learner: mask.sum() == 0 at t_env {}".format(t_env))
            return

        # hc修改
        # critic_mask = mask.clone()
        # mask = mask.repeat(1, 1, self.n_agents)
        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        # old_mac_out = []
        old_mac_out_1 = []
        old_mac_out_2 = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs_1, agent_outs_2 = self.old_mac.forward(batch, t=t)  # 计算整个序列的
            # old_mac_out.append(agent_outs)
            old_mac_out_1.append(agent_outs_1)
            old_mac_out_2.append(agent_outs_2)
        # 按照每个step进行append
        # old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        # old_pi = old_mac_out
        # old_pi[mask == 0] = 1.0
        # head 1
        old_mac_out_1 = th.stack(old_mac_out_1, dim=1)  # Concat over time ,然后通过th.stack在dim=1进行叠加得到(batch_size, max_length, num_agents, action_space)
        old_pi_1 = old_mac_out_1
        old_pi_1[mask == 0] = 1.0
        # head 2
        old_mac_out_2 = th.stack(old_mac_out_2, dim=1)
        old_pi_2 = old_mac_out_2
        old_pi_2[mask == 0] = 1.0

        # old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        # old_log_pi_taken = th.log(old_pi_taken + 1e-10)
        # head 1
        old_pi_taken_1 = th.gather(old_pi_1, dim=3, index=actions_1).squeeze(3)   # 根据actions_1选择相应的概率，并通过squeeze(3)去除第三纬度，最后得到(2, 657, 5)
        old_log_pi_taken_1 = th.log(old_pi_taken_1 + 1e-10)
        # old_log_pi_taken_1 是指在mask为0的位置上output为1，那么求log之后，就是为0
        # head 2
        old_pi_taken_2 = th.gather(old_pi_2, dim=3, index=actions_2).squeeze(3)
        old_log_pi_taken_2 = th.log(old_pi_taken_2 + 1e-10)


        for k in range(self.args.epochs):
            # mac_out = []
            mac_out_1 = []
            mac_out_2 = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs_1, agent_outs_2 = self.mac.forward(batch, t=t)
                # mac_out.append(agent_outs)
                mac_out_1.append(agent_outs_1)
                mac_out_2.append(agent_outs_2)

            # mac_out = th.stack(mac_out, dim=1)  # Concat over time
            mac_out_1 = th.stack(mac_out_1, dim=1)
            mac_out_2 = th.stack(mac_out_2, dim=1)

            # pi = mac_out
            pi_1 = mac_out_1
            pi_2 = mac_out_2

            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                          critic_mask)
            # advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.critic, batch, rewards,
            #                                                               critic_mask)

            advantages = advantages.detach()

            # Calculate policy grad with mask

            # pi[mask == 0] = 1.0
            pi_1[mask == 0] = 1.0
            pi_2[mask == 0] = 1.0

            # head 1 loss
            pi_taken_1 = th.gather(pi_1, dim=3, index=actions_1).squeeze(3)
            log_pi_taken_1 = th.log(pi_taken_1 + 1e-10)

            ratios_1 = th.exp(log_pi_taken_1 - old_log_pi_taken_1.detach())
            surr1_1 = ratios_1 * advantages
            surr2_1 = th.clamp(ratios_1, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy_1 = -th.sum(pi_1 * th.log(pi_1 + 1e-10), dim=-1)
            pg_loss_1 = -((th.min(surr1_1, surr2_1) + self.args.entropy_coef * entropy_1) * mask).sum() / mask.sum()

            # head 1 loss
            pi_taken_2 = th.gather(pi_2, dim=3, index=actions_2).squeeze(3)
            log_pi_taken_2 = th.log(pi_taken_2 + 1e-10)

            ratios_2 = th.exp(log_pi_taken_2 - old_log_pi_taken_2.detach())
            surr1_2 = ratios_2 * advantages
            surr2_2 = th.clamp(ratios_2, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy_2 = -th.sum(pi_2 * th.log(pi_2 + 1e-10), dim=-1)
            pg_loss_2 = -((th.min(surr1_2, surr2_2) + self.args.entropy_coef * entropy_2) * mask).sum() / mask.sum()

            pg_loss = pg_loss_1 + pg_loss_2
            # pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            # log_pi_taken = th.log(pi_taken + 1e-10)
            #
            # ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            # surr1 = ratios * advantages
            # surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages
            #
            # entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            # pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            print(f'policy gradient loss: {pg_loss}')
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            # self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pi1_max", (pi_1.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pi2_max", (pi_2.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            # hc修改
            # target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }
        # hc
        # v = critic(batch)[:, :-1]
        v = critic(batch)[:, :-1].squeeze(3)  # ippo
        # v = th.repeat_interleave(v, self.args.n_agents, dim=2)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        print(f'critic loss: {loss}')

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        rewards_ = th.repeat_interleave(rewards, self.args.n_agents, dim=2)
        # values_ = th.repeat_interleave(values, self.args.n_agents, dim=2)
        # rewards_ = rewards
        values_ = values.squeeze(3)
        nstep_values = th.zeros_like(values_[:, :-1])
        for t_start in range(rewards_.size(1)):
            nstep_return_t = th.zeros_like(values_[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards_.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values_[:, t] * mask[:, t]
                elif t == rewards_.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards_[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values_[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards_[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
