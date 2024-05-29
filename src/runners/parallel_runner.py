from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        # env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        ip_base = '127.0.0.1'
        port_base = 15663
        for i, worker_conn in enumerate(self.worker_conns):
            new_port = port_base + ((i+1)*2)
            new_ip = ip_base + ':' + str(new_port)
            print(f"Curr runner IP is: {new_ip}")
            env_class = env_REGISTRY[self.args.env]
            # env_instance = env_class(logger=self.logger, address=new_ip, **self.args.env_args)
            wrapped_env_instance = CloudpickleWrapper(partial(env_class, logger=self.logger, address=new_ip, **self.args.env_args))

            ps = Process(target=env_worker,
                         args=(worker_conn, wrapped_env_instance))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        # pre_transition_data = {
        #     "state": [],
        #     "avail_actions": [],
        #     "obs": []
        # }
        pre_transition_data = {
            "state": [],
            "avail_actions_1": [],
            "avail_actions_2": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            # pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["avail_actions_1"].append(data["avail_actions_1"])
            pre_transition_data["avail_actions_2"].append(data["avail_actions_2"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False, cur_time=None):
        self.reset()
        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # if save_probs:
            #     actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            # else:
            actions_1, actions_2 = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)

            # re_actions_1 = [[] for _ in range(self.batch_size)]
            # re_actions_2 = [[] for _ in range(self.batch_size)]

            # construct actions with envs_not_terminated
            # for i, idx in enumerate(envs_not_terminated):
            #     re_actions_1[idx] = actions_1[i].tolist()
            #     re_actions_2[idx] = actions_2[i].tolist()
            #
            # re_actions_1 = th.tensor(re_actions_1)
            # re_actions_2 = th.tensor(re_actions_2)

            # cpu_actions = actions.to("cpu").numpy()
            # cpu_actions_1 = re_actions_1.to("cpu").numpy()
            # cpu_actions_2 = re_actions_2.to("cpu").numpy()

            cpu_actions_1 = actions_1.to("cpu").numpy()   # (batch_size, 5)
            cpu_actions_2 = actions_2.to("cpu").numpy()

            # (batch_size_run, num_agents, action_space)
            batch_action_data = []
            # for run_idx in range(self.batch_size):
            # for run_idx in range(len(cpu_actions_1)):
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                # All parent_conn
                agent_actions = [[] for _ in range(5)]  # agents_num
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        for i in range(5):
                            try:
                                agent_actions[i].append(cpu_actions_1[action_idx][i])  #
                                agent_actions[i].append(cpu_actions_2[action_idx][i])
                            except Exception as e:
                                print(f'Error: {e}')
                                print(f'action_idx: {action_idx}')
                                print(f'i: {i}')
                                print(f'cpu_actions_1: {cpu_actions_1}')
                                print(f'envs_not_terminated')
                    action_idx += 1
                batch_action_data.append(agent_actions)  # (batch_size, )


            # Update the actions taken
            actions_chosen = {
                # "actions": actions.unsqueeze(1).to("cpu"),
                "actions_1": actions_1.unsqueeze(1).to("cpu"),
                "actions_2": actions_2.unsqueeze(1).to("cpu"),
            }
            # if save_probs:
            #     actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)


            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        # parent_conn.send(("step", cpu_actions[action_idx]))
                        # print(f'idx: {idx}')
                        # print(f'Send actions: {batch_action_data}')
                        # print(envs_not_terminated)
                        # print(terminated)
                        parent_conn.send(("step", batch_action_data[action_idx]))  # list index out of range:
                action_idx += 1  # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                # "avail_actions": [],
                "avail_actions_1": [],
                "avail_actions_2": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()  # receive from env worker
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    # pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["avail_actions_1"].append(data["avail_actions_1"])
                    pre_transition_data["avail_actions_2"].append(data["avail_actions_2"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch, None

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data  #
            # actions what shape?
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            # avail_actions = env.get_avail_actions()
            available_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                # "avail_actions": avail_actions,
                "avail_actions_1": [available_actions['0']],
                "avail_actions_2": [available_actions['1']],
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,  # terminated != env_info.get("episode_limit", False), # terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            available_actions = env.get_avail_actions()
            remote.send({
                "state": env.get_state(),
                # "avail_actions": env.get_avail_actions(),
                "avail_actions_1": [available_actions['0']],
                "avail_actions_2": [available_actions['1']],
                "obs": env.get_obs()
            })
        elif cmd == "close":
            # env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

