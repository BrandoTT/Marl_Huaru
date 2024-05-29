from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import os
import json

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](logger=self.logger, address="127.0.0.1:53453", **self.args.env_args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0
        self.episode_total = 0
        
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close_environ()

    def reset(self, if_test=None, cur_time=None):
        self.batch = self.new_batch()
        if if_test==None:
            self.env.reset(args=self.args, cur_time=cur_time)
        else:
            self.env.reset(if_test=if_test, args=self.args, cur_time=cur_time)

        self.t = 0

    def run(self, test_mode=False, cur_time=None):
        # monster_hp_json_file_path = os.path.join('/workspace/Userlist/hucheng/pycharm/Hok_Marl_ppo/results/sacred/hok/monster_last_hp/', f'{cur_time}_monster_lasthp_{self.args.name}.txt')
        if self.args.env == 'hok':
            self.reset(if_test=test_mode, cur_time=cur_time)
        else:
            self.reset(cur_time=cur_time)

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            available_actions = self.env.get_avail_actions()
            pre_transition_data = {
                "state": [self.env.get_state()],
                # "avail_actions": [self.env.get_avail_actions()],
                "avail_actions_1": [available_actions['0']],
                "avail_actions_2": [available_actions['1']],
                "obs": [self.env.get_obs()]
            }
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # actions 是(1, num_agents)的数组，代表每个Agent最后选择的动作
            # actions_1是head1的动作，actions_2是head2的动作
            actions_1, actions_2 = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # actions_1[0] what that mean???

            agent_actions = [[] for _ in range(self.env.n_agents)]
            for i in range(len(actions_1[0])):
                agent_actions[i].append(actions_1[0][i])
                agent_actions[i].append(actions_2[0][i])

            # agent_actions 是 [[3, 4], [4, 5], [3, 5], [2, 0], [2, 0]] 是每个Agent两个head动作输出
            # actions = actions[0] # for ippo
            # Fix memory leak
            # cpu_actions = actions[0].to("cpu").numpy()
            cpu_actions_1 = actions_1[0].to("cpu").numpy()
            cpu_actions_2 = actions_2[0].to("cpu").numpy()

            # 这是跟环境交互的最重要的一步
            reward, terminated, env_info = self.env.step(agent_actions, if_test=test_mode)
            episode_return += reward
            post_transition_data = {
                # "actions": cpu_actions,
                "actions_1": cpu_actions_1,
                "actions_2": cpu_actions_2,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],  # 为什么是这样设计？？
                # terminated如果是True，说明还没有到episode_limit,但是提前结束。如果是False，则说明到了limit，表示还未结束。
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.t += 1
            # print(f'receive: {terminated}')
            # print(f'info: {(env_info.get("episode_limit", False),)}')
            # print('terminated', post_transition_data["terminated"])
        # last_data = {
        #     "state": [self.env.get_state()],
        #     # "avail_actions": [self.env.get_avail_actions()],
        #     "avail_actions_1": [self.env.get_avail_actions()['0']],
        #     "avail_actions_2": [self.env.get_avail_actions()['1']],
        #     "obs": [self.env.get_obs()]
        # } # last_data应该没有作用，这个是充数的
        available_actions = self.env.get_avail_actions()
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions_1": [available_actions['0']],
            "avail_actions_2": [available_actions['1']],
            "obs": [pre_transition_data["obs"][0]]
        }  # last_data应该没有作用，这个是充数的

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions_1, actions_2 = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions_1 = actions_1[0].to("cpu").numpy()
        cpu_actions_2 = actions_2[0].to("cpu").numpy()

        self.batch.update({"actions_1": cpu_actions_1, "actions_2": cpu_actions_2}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t
            self.episode_total += 1

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, env_info

    def _log(self, returns, stats, prefix):
        # stats: {'battle_won': 0, 'n_episodes': 1, 'ep_length': 45} battle_won从哪里来的
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
