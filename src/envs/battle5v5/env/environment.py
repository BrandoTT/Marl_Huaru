"""
!/usr/bin/python3
-*- coding: utf-8 -*-
@FileName: environment.py
@Time: 2024/4/15 下午3:54
@Author: ZhengtaoCao
@Description: None
"""

from .env_runner import EnvRunner
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch as th
from time import sleep
from ..config import ADDRESS, config, POOL_NUM, ISHOST, XSIM_NUM
# from .multiagentenv import MultiAgentEnv
import math
import logging

logger = logging.getLogger('Battle_5v5')
logger.setLevel(logging.DEBUG)

BLUE_INFO = {
    '0': {'Name': '蓝有人机', 'ID': 6},
    '1': {'Name': '蓝无人机1', 'ID': 14},
    '2': {'Name': '蓝无人机2', 'ID': 15},
    '3': {'Name': '蓝无人机3', 'ID': 16},
    '4': {'Name': '蓝无人机4', 'ID': 17},
}
RED_INFO = {
    '0': {'Name': '红有人机', 'ID': 1},
    '1': {'Name': '红无人机1', 'ID': 2},
    '2': {'Name': '红无人机2', 'ID': 11},
    '3': {'Name': '红无人机3', 'ID': 12},
    '4': {'Name': '红无人机4', 'ID': 13},
}
BLUE_FIRE_INFO = {
    '0': {'Name': '空空导弹_1(蓝有人机_武器系统_1)', 'ID': 2147483668},
    '1': {'Name': '空空导弹_2(蓝有人机_武器系统_1)', 'ID': 2147483669},
    '2': {'Name': '空空导弹_3(蓝有人机_武器系统_1)', 'ID': 2147483670},
    '3': {'Name': '空空导弹_4(蓝有人机_武器系统_1)', 'ID': 2147483671},
    '4': {'Name': '空空导弹_1(蓝无人机1_武器系统_1)', 'ID': 2147483657},
    '5': {'Name': '空空导弹_2(蓝无人机1_武器系统_1)', 'ID': 2147483659},
    '6': {'Name': '空空导弹_1(蓝无人机2_武器系统_1)', 'ID': 2147483652},
    '7': {'Name': '空空导弹_2(蓝无人机2_武器系统_1)', 'ID': 2147483654},
    '8': {'Name': '空空导弹_1(蓝无人机3_武器系统_1)', 'ID': 2147483660},
    '9': {'Name': '空空导弹_2(蓝无人机3_武器系统_1)', 'ID': 2147483662},
    '10': {'Name': '空空导弹_1(蓝无人机4_武器系统_1)', 'ID': 2147483648},
    '11': {'Name': '空空导弹_2(蓝无人机4_武器系统_1)', 'ID': 2147483650},
}

class HuaRuBattleEnvWrapper(EnvRunner):
    def __init__(self,
                 map_name='battle5v5',
                 num_agents=5,
                 time_step=0,
                 seed=123,
                 episode_limit=2000,
                 agents=config['agents'],
                 address=None,  # ADDRESS['ip'] + ":" + str(ADDRESS['port']),
                 mode='host',
                 logger=logger
        ):
        EnvRunner.__init__(self, agents, address, mode)
        # MultiAgentEnv.__init__(self)
        self.map_name = map_name
        self.address = address
        self.n_agents = num_agents

        # 动作空间
        self.action_space_1 = 10  # 9个方向
        self.action_space_2 = 6  # 5个攻击动作加不攻击
        # 状态空间
        self.observation_space = 124  # [spaces.Box(low=-1, high=1, shape=(40,))]  # shapes1.box
        # 初始化智能体，红方智能体是用于转换仿真的指令；蓝方智能体适用于利用代码规则
        # 记录红蓝方的每帧的states
        self.ori_message = None
        self.episode_limit = episode_limit
        self.seed = seed
        self.red_death = None  # 存入红方死亡的实体ID
        self.blue_death = None  # 存入蓝方死亡实体ID

        # 记录每一帧每个飞机实体的位置
        self.red_agent_loc = None

        # 存储上一帧的原始战场态势数据
        self.last_ori_message = None
        # 存取本episode训练的其他信息
        self.info = None
        self.forward_step = None
        self.logger = logger
        self.done = None

        self.reward = None
        self.shoot_interval = None  # 每个Agent的设计间隔
        self.shoot_interval_step = 20  # 50000米发射，导弹1000m/s

        # 记录胜率
        self.battles_won = 0.
        self.battles_game = 0.

    def reset(self, if_test=False, args=None, cur_time=None):
        """重置仿真环境, 返回初始帧obs"""
        for side, agent in self.agents.items():
            agent.reset()
        super().reset()

        self.logger.console_logger.info('Reset, Restart Engine!')
        self.forward_step = 0
        self.red_death = []
        self.blue_death = []
        self.red_agent_loc = {'0': None, '1': None, '2': None, '3': None, '4': None}
        self.info = {'Battle Won': 2}  # 2代表是平局, 0红方胜利，1蓝方胜利
        self.obs = None
        self.ori_message = None
        self.shoot_interval = np.zeros([self.n_agents, 5])

        # 重置奖励函数
        self.reward = 0.
        # 重置训练环境

        self.ori_message = super().step([])  # 推动拿到第一帧的obs信息

        cmd_list = []
        cmd_list.extend(self.agents["red"].make_init_cmd())

        self.last_ori_message = self.ori_message
        self.ori_message = super().step(cmd_list)

        # 获取RL Obs & State
        self.obs = self.get_obs()
        self.state = self.get_state()

        return self.obs, self.state   # list, np

    def step(self, actions=None, if_test=False):
        """
            逐帧推进仿真引擎, 发送actions, 返回obs
            :param actions: [[0, 2],
                            [1, 3],
                            ...
                            ] 是一个二维数组，规模是(5, 2) 第1列是移动动作，动作空间是9；第2列是攻击动作，动作空间是6
        """
        for index in range(len(actions)):
            # 拿取每个agent的攻击动作
            cur_shoot_action = actions[index][1]
            if cur_shoot_action == 5:
                pass
            else:
                self.shoot_interval[index][cur_shoot_action] = self.shoot_interval_step

        non_zero_indices = self.shoot_interval != 0
        self.shoot_interval[non_zero_indices] -= 1
        self.forward_step += 1

        if_done = False
        # 准备将自己已经知道的所有信息一起发送给Agent
        parse_msg_ = {'agent_pre_loc': self.red_agent_loc,
                     'blue_info': BLUE_INFO,
                     'red_info': RED_INFO
        }

        # 生成蓝方Agents的cmd list
        blue_cmd_list = self.get_blue_cmd(self.agents["blue"])
        # 将网络输出动作转换为仿真执行指令,给self.red_agents
        cmd_list = self.agents["red"].make_actions(actions, parse_msg_) # Agents的仿真指令
        cmd_list.extend(blue_cmd_list)
        # print(f'当前生成的指令包括：{cmd_list}')
        self.last_ori_message = self.ori_message
        # 将仿真指令，发送回仿真，并且拿到下一帧的状态
        self.ori_message = super().step(cmd_list)
        # print(super().get_done(self.ori_message))
        self.done, flag_no = super().get_done(self.ori_message)

        # 解析得到下一帧的Agents Obs
        self.obs = self.get_obs()
        self.state = self.get_state()  # 更新全局state

        information = {"battle_win": False}

        if self.done[0] or self.forward_step >= self.episode_limit:  # 或者步数超过episode_limit
            if not self.done[0]:
                # 超过了步数
                # 说明当前episode的训练已经结束
                self.logger.console_logger.info('当前Episode训练已经结束(超过指定训练步数)！')
                self.info['Battle Won'] = 2
                self.logger.console_logger.info("平  局")
                # 结束的时候不需要kill env，只是下个episode开始的时候，reset()一下Agents就可以了
                information["episode_limit"] = True
                if_done = True
                # self.close_environ()
                # super().reset()

            else:
                # 说明当前episode的训练已经结束
                self.logger.console_logger.info('当前Episode训练已经结束！')
                if (self.done[1] == 1 and self.done[2] == 0):
                    self.logger.console_logger.info("红 方 胜!")
                    self.info['Battle Won'] = 0
                    information['battle_win'] = True  # self.get_stats()['win_rate']
                    self.battles_won += 1
                elif (self.done[1] == 0 and self.done[2] == 1):
                    self.logger.console_logger.info("蓝 方 胜!")
                    self.info['Battle Won'] = 1
                else:
                    self.logger.console_logger.info("平  局")
                    self.info['Battle Won'] = 2
                # 结束的时候不需要kill env，只是下个episode开始的时候，reset()一下Agents就可以了
                if_done = True
                # self.close_environ()
                # super().reset()

            self.battles_game += 1
            # self.logger.console_logger.info(f'取得的胜率：{self.get_stats()["win_rate"]}')
            # 计算红方有人，无人
            red_total_num = len(self.ori_message['red']['platforminfos'])
            if 1 in self.red_death:
                red_av_num = 0
                red_uav_num = red_total_num
            else:
                red_av_num = 1
                red_uav_num = red_total_num - 1

            blue_total_num = len(self.ori_message['blue']['platforminfos'])
            if 6 in self.blue_death:
                blue_av_num = 0
                blue_uav_num = blue_total_num
            else:
                blue_av_num = 1
                blue_uav_num = blue_total_num - 1

            self.logger.console_logger.info(
                f"红方剩余兵力：有人机剩余 {red_av_num} 架，无人机剩余: {red_uav_num} 架; 红方剩余弹药情况：{sum([entity['LeftWeapon'] for entity in self.ori_message['red']['platforminfos']])}")
            self.logger.console_logger.info(
                f"蓝方剩余兵力：有人机剩余 {blue_av_num} 架，无人机剩余: {blue_uav_num} 架; 蓝方剩余弹药情况：{sum([entity['LeftWeapon'] for entity in self.ori_message['blue']['platforminfos']])}")

        cur_reward = self.get_reward_adjust(self.last_ori_message, self.ori_message, if_done, flag_no)
        # cur_reward = self.get_reward_test(self.last_ori_message, self.ori_message, if_done, flag_no)
        # self.reward计算的是累积奖励，没有什么用, cur_reward才是当前帧的奖励
        self.reward += cur_reward

        if self.done[0]:
            self.logger.console_logger.info(f'本Episode获得的最终奖励是: {cur_reward}')
            self.logger.console_logger.info(f'本Episode获得的累积奖励: {self.reward}')

        if self.forward_step % 400 == 0:
            self.logger.console_logger.info('Forward step: %d' % self.forward_step)
            self.logger.console_logger.info(f'Cur frame reward: {cur_reward}')
            self.logger.console_logger.info(f'Total reward: {self.reward}')

        # 结束环境
        return cur_reward, if_done, information  # self.info

    def get_blue_cmd(self, blue_agents):
        """获取蓝方当前的cmd_list"""
        cmd_list = []
        cur_time = self.ori_message["sim_time"]
        cmd_list = super()._agent_step(blue_agents, cur_time, self.ori_message["blue"]) + cmd_list

        return cmd_list

    def get_reward(self, last_obs=None, next_obs=None, if_done=False):
        """
            每帧计算reward值，使用团队奖励，当敌方的某个战机被击落后，产生正奖励，当己方的一个飞机被击落，产生负奖励；
            到最后，即if_done==True，说明episode结束，开始进行结局奖励计算。
            :param last_obs: 上一帧的战场局势(原始数据形式)
            :param next_obs: 下一帧的战场局势(原始数据形式)
            :param if_done: 本episode是否已经结束？如果结束的话，要进行战场判定
        """
        reward = 0.
        """战场没有结束，只需要统计占损奖励"""
        # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
        last_red_agent_num = len(last_obs['red']['platforminfos'])
        last_red_weapon_num = 0.
        for entity in last_obs['red']['platforminfos']:
            last_red_weapon_num += entity['LeftWeapon']
        # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
        next_red_agent_num = len(next_obs['red']['platforminfos'])
        next_red_weapon_num = 0.
        for entity in next_obs['red']['platforminfos']:
            next_red_weapon_num += entity['LeftWeapon']
        # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        last_blue_agent_num = len(last_obs['blue']['platforminfos'])
        last_blue_weapon_num = 0.
        for entity in last_obs['blue']['platforminfos']:
            last_blue_weapon_num += entity['LeftWeapon']
        # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        next_blue_agent_num = len(next_obs['blue']['platforminfos'])
        next_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            next_blue_weapon_num += entity['LeftWeapon']

        # 计算占损战耗奖励
        reward = (-1.2) * (10 * (last_red_agent_num - next_red_agent_num)) - 0.5 * (last_red_weapon_num - next_red_weapon_num) \
                 + 1.2 * (10 * (last_blue_agent_num - next_blue_agent_num) + 0.5 * (last_blue_weapon_num - next_blue_weapon_num))
        # reward = (-1.5) * (10 * (last_red_agent_num - next_red_agent_num)) \
        #          + 2.0 * (50 * (last_blue_agent_num - next_blue_agent_num) + 0.5 * (last_blue_weapon_num - next_blue_weapon_num))

        # print(f'reward1: {reward}')
        # if last_red_agent_num - next_red_agent_num == 0:
        #     reward += 1  # 如果数量能保持现状，不死那么+5
        # print(f'reward2: {reward}')
        distance_reward = 0.
        if not if_done:
            # 设置一个奖励：计算每一帧红方每个飞机离中心点的距离，距离中心点越近得分越高
            for agent_order in range(len(self.ori_message['red']['platforminfos'])):
                cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
                if cur_id in self.red_death:
                    distance_reward += 0.
                else:
                    # 当前的坐标
                    cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
                    cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
                    cur_z = self.ori_message['red']['platforminfos'][agent_order]['Alt']
                    cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2 + cur_z ** 2)
                    # 上一个坐标
                    last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
                    last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
                    last_z = self.last_ori_message['red']['platforminfos'][agent_order]['Alt']
                    last_distance = math.sqrt(last_x ** 2 + last_y ** 2 + last_z ** 2)

                distance_reward += 0.002 if (last_distance - cur_distance) >= 0 else -0.002

        else:
            """战场已经结束, 需要额外统计结局奖励"""
            if self.info['Battle Won'] == 0: # 红方胜利
                reward += 600
            elif self.info['Battle Won'] == 1:
                reward -= 600
            else:
                reward += 0

        reward += distance_reward

        return reward

    def get_reward_adjust(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):
        """
            每帧计算reward值，使用团队奖励，当敌方的某个战机被击落后，产生正奖励，当己方的一个飞机被击落，产生负奖励；
            到最后，即if_done==True，说明episode结束，开始进行结局奖励计算。
            :param last_obs: 上一帧的战场局势(原始数据形式)
            :param next_obs: 下一帧的战场局势(原始数据形式)
            :param if_done: 本episode是否已经结束？如果结束的话，要进行战场判定
        """
        reward = 0.  # 最终奖励
        distance_reward = 0.  # 距离奖励, 距离中心点越近越胜利，越来越低
        attack_reward = 0.  # 攻击奖励，攻击敌方越多伤害越高，毁伤奖励应该是越来越高的
        damaged_reward = 0.  # 毁伤奖励， 自己造成的毁伤越多得到的负奖励越高

        """战场没有结束，只需要统计占损奖励"""
        # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
        last_red_agent_num = len(last_obs['red']['platforminfos'])
        last_red_weapon_num = 0.
        for entity in last_obs['red']['platforminfos']:
            last_red_weapon_num += entity['LeftWeapon']
        # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
        next_red_agent_num = len(next_obs['red']['platforminfos'])
        next_red_weapon_num = 0.
        for entity in next_obs['red']['platforminfos']:
            next_red_weapon_num += entity['LeftWeapon']
        # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        last_blue_agent_num = len(last_obs['blue']['platforminfos'])
        last_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            last_blue_weapon_num += entity['LeftWeapon']
        # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        next_blue_agent_num = len(next_obs['blue']['platforminfos'])
        next_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            next_blue_weapon_num += entity['LeftWeapon']

        # 计算伤害奖励
        if last_red_agent_num - next_red_agent_num > 0:
            # 说明上一轮红方飞机更多，红方飞机有伤亡
            if RED_INFO['0']['ID'] in self.red_death:
                # 红方有人机被打掉，红方输掉了
                damaged_reward -= 12
            else:
                # 红方无人机被打掉，减小分
                damaged_reward = damaged_reward - (2 * (last_red_agent_num - next_red_agent_num))
                # damaged_reward -= 200 * (last_red_agent_num - next_red_agent_num)  # 地方是不是不对，如果同一帧，死掉两个实体呢？这里应该是 *
        else:
            # 说明红方飞机数量没有变化
            damaged_reward -= 0.

        reward += damaged_reward

        # 计算攻击奖励
        if last_blue_agent_num - next_blue_agent_num > 0:
            # 说明上一帧蓝方飞机要更多
            if BLUE_INFO['0']['ID'] in self.blue_death:
                attack_reward += 12
            else:
                attack_reward += 2 * (last_blue_agent_num - next_blue_agent_num)
        else:
            # 说明蓝方飞机数量没有变化
            attack_reward += 0.

        reward += attack_reward

        # 计算距离奖励
        for agent_order in range(len(self.ori_message['red']['platforminfos'])):
            cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
            if cur_id in self.red_death:
                distance_reward += 0.
            else:
                # 当前的坐标
                cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
                cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
                cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2)
                # 上一个坐标
                last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
                last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
                last_distance = math.sqrt(last_x ** 2 + last_y ** 2)
                # 这个值是很大的
                # distance_reward += 1 if (last_distance - cur_distance) > 0 else -1
                distance_reward += 0.001 if (last_distance - cur_distance) > 0 else -0.001

        reward += (distance_reward * self.decay_coefficient(step=self.forward_step))  # 1.0是衰减系数

        return reward

    def get_reward_test(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):

        reward = 0.
        victory_reward = 0.
        distance_reward = 0.
        attack_reward = 0.
        damage_reward = 0.
        missile_reward = 0.
        protection_reward = 0.
        center_radius = 40000

        # 1. 统计上一帧中，红方战机的数量 & 存在的导弹剩余数量
        last_red_agent_num = len(last_obs['red']['platforminfos'])
        last_red_weapon_num = 0.
        for entity in last_obs['red']['platforminfos']:
            last_red_weapon_num += entity['LeftWeapon']
        # 2. 统计下一帧中，红方战机的数量 & 存在的导弹剩余数量
        next_red_agent_num = len(next_obs['red']['platforminfos'])
        next_red_weapon_num = 0.
        for entity in next_obs['red']['platforminfos']:
            next_red_weapon_num += entity['LeftWeapon']
        # 3. 统计上一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        last_blue_agent_num = len(last_obs['blue']['platforminfos'])
        last_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            last_blue_weapon_num += entity['LeftWeapon']
        # 4. 统计下一帧中，蓝方战机的数量 & 存在的导弹剩余数量
        next_blue_agent_num = len(next_obs['blue']['platforminfos'])
        next_blue_weapon_num = 0.
        for entity in next_obs['blue']['platforminfos']:
            next_blue_weapon_num += entity['LeftWeapon']

        # victory reward
        if if_done:
            if self.info['Battle Won'] == 0:  # 红方胜利
                # remaining_missiles = sum([entity['LeftWeapon'] for entity in next_obs['red']['platforminfos']])
                victory_reward = 40  # + 2 * remaining_missiles
                print('红方胜利', victory_reward)
            elif self.info['Battle Won'] == 1:  # 蓝方胜利
                used_missiles = sum([entity['LeftWeapon'] for entity in next_obs['red']['platforminfos']])
                # used_missiles = 12 - used_missiles
                victory_reward = -40  # - 2 * used_missiles
                print('蓝方胜利', victory_reward)
            else:  # 平局
                victory_reward = 0

        reward += victory_reward

        # distance_reward
        for agent_order in range(len(self.ori_message['red']['platforminfos'])):
            cur_id = self.ori_message['red']['platforminfos'][agent_order]['ID']
            if cur_id in self.red_death:
                distance_reward += 0.
            else:
                # 当前的坐标
                cur_x = self.ori_message['red']['platforminfos'][agent_order]['X']
                cur_y = self.ori_message['red']['platforminfos'][agent_order]['Y']
                cur_distance = math.sqrt(cur_x ** 2 + cur_y ** 2)  # 计算当前距离

                # 上一个坐标
                last_x = self.last_ori_message['red']['platforminfos'][agent_order]['X']
                last_y = self.last_ori_message['red']['platforminfos'][agent_order]['Y']
                last_distance = math.sqrt(last_x ** 2 + last_y ** 2)  # 计算上一帧的距离

                if cur_distance < center_radius:  # 如果红方单位在中心区域内
                    distance_reward += 1 / 500  # 每多存活一帧，得到一个奖励
                else:  # 如果红方单位在中心区域外
                    if cur_distance < last_distance:
                        distance_reward += 1 / 500  # 向中心区域移动，得到正向奖励
                    else:
                        distance_reward += -1 / 500  # 远离中心区域，得到负向奖励

        reward += distance_reward

        # attack_reward
        if last_blue_agent_num - next_blue_agent_num > 0:
            # 说明上一帧蓝方飞机要更多
            if BLUE_INFO['0']['ID'] in self.blue_death:
                attack_reward += 50
            #        print('蓝方有人机被击落', attack_reward)
            else:
                attack_reward += 10 * (last_blue_agent_num - next_blue_agent_num)
        #        print('蓝方wu人机被击落', attack_reward)
        else:
            # 说明蓝方飞机数量没有变化
            attack_reward += 0.

        reward += attack_reward

        if last_red_agent_num - next_red_agent_num > 0:
            # 说明上一帧红方飞机要更多
            if RED_INFO['0']['ID'] in self.red_death:
                damage_reward -= 50
            #        print('红方有人机被击落', damage_reward)
            else:
                damage_reward -= 10 * (last_red_agent_num - next_red_agent_num)
        #        print('红方wu人机被击落', damage_reward)
        else:
            # 说明蓝方飞机数量没有变化
            damage_reward += 0.

        reward += damage_reward

        # missile reward
        last_red_missile_num = sum([entity['LeftWeapon'] for entity in last_obs['red']['platforminfos']])
        next_red_missile_num = sum([entity['LeftWeapon'] for entity in next_obs['red']['platforminfos']])
        missile_reward += (last_red_missile_num - next_red_missile_num) * -1

        reward += missile_reward

        # protection_reward
        if not if_done:
            # 获取有人机的位置
            manned_aircraft_position = None
            manned_aircraft_heading = None
            for entity in self.ori_message['red']['platforminfos']:
                if entity['ID'] == RED_INFO['0']['ID']:  # 假设有人机的ID为RED_INFO中的'0'
                    manned_aircraft_position = [entity['X'], entity['Y'], entity['Alt']]
                    manned_aircraft_heading = entity['Heading']
                    break

            if manned_aircraft_position is not None and manned_aircraft_heading is not None:
                # 计算无人机与有人机的距离，并根据距离计算保护奖励
                for i in range(1, 5):
                    unmanned_aircraft_position = None
                    for entity in self.ori_message['red']['platforminfos']:
                        if entity['ID'] == RED_INFO[str(i)]['ID']:
                            unmanned_aircraft_position = [entity['X'], entity['Y'], entity['Alt']]
                            break

                    if unmanned_aircraft_position is not None:
                        distance = self.distance_computation(manned_aircraft_position, unmanned_aircraft_position)
                        # 假设保护距离阈值为40000
                        if distance < 40000:
                            protection_reward += 1.0 / 500
                        else:
                            protection_reward -= 1.0 / 500
                        # 如果有人机在无人机前面，给予负向奖励

        reward += protection_reward
        return reward


    def decay_coefficient(self, step=None):
        total_steps = self.episode_limit
        # total_steps = self.episode_limit
        decay_rate = -np.log(0.01) / total_steps  # 衰减速率，这里假设最终值为0.01
        return np.exp(-decay_rate * step)

    def get_simple_reward(self, last_obs=None, next_obs=None, if_done=False, flag_no=False):
        """获取一个简单奖励，胜利：1，平局：0，失败：-1

        :param last_obs: 上一局的态势
        :param next_obs: 本局的态势
        :param if_done: 是否本episode已经结束
        :param flag_no: 用来判断是否红方是通过长时间占领中心区域获胜的

        :return reward: float
        """
        reward = 0.
        if not if_done:
            pass
        else:
            if self.info['Battle Won'] == 0: # 红方胜利
                reward = 1
                if flag_no:
                    reward = 0
            elif self.info['Battle Won'] == 1:
                reward = -1
            else:
                reward = 0

        return reward

    def get_obs(self):
        """获取所有Agent的obs = [obs1, obs2, ..., obs5]"""
        agent_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agent_obs

    def get_obs_agent(self, agent_order):
        """获取每个Agent自己的obs"""
        # 已经有一个全局属性self.ori_message
        assert self.ori_message is not None
        # 通过解析这个原始数据构造每个Agent的obs，先通过这个id来判断是否已经死亡，就是判断字典中是否有这个实体
        self.cur_agent_name = RED_INFO[str(agent_order)]['Name']  # 当前Agent的名字
        self.cur_agent_id = RED_INFO[str(agent_order)]['ID']  # 当前Agent的id
        # 查找返回的信息中是否包含当前Agent，如果没有的话说明当前Agent已经死亡
        cur_agent_exist = any(filter(lambda x: x['ID'] == self.cur_agent_id, self.ori_message['red']['platforminfos']))
        if not cur_agent_exist:
            """说明这个Agent已经不存在了"""
            self.red_agent_loc[str(agent_order)] = None
            self.red_death.append(self.cur_agent_id)
        """当前Agent的obs"""
        agent_obs = self.parse_msg(str(agent_order), cur_agent_exist)
        return np.array(agent_obs)

    def parse_msg(self, agent_order, exist=None):
        """对原始态势信息进行解析处理，构造状态"""
        cur_agent_obs = []
        platforminfos_list = self.ori_message['red']['platforminfos']
        if exist:
            """当前Agent还存活"""
            # 获取当前Agent在['platforminfos']中的索引
            cur_index = next(((i, item) for i, item in enumerate(platforminfos_list) if item['ID'] == RED_INFO[str(agent_order)]['ID']), (-1, None))[0]
            ## 拿这个Agent本身的信息
            identification_own = 0 if platforminfos_list[cur_index]['Identification'] == '红方' else 1
            type_own = platforminfos_list[cur_index]['Type']
            availability_own =platforminfos_list[cur_index]['Availability']
            x_own = platforminfos_list[cur_index]['X']
            y_own = platforminfos_list[cur_index]['Y']
            alt_own = platforminfos_list[cur_index]['Alt']
            z_own = alt_own
            accmag_own = platforminfos_list[cur_index]['AccMag']
            heading_own = platforminfos_list[cur_index]['Heading']
            # new add
            pitch_own = platforminfos_list[cur_index]['Pitch']
            roll_own = platforminfos_list[cur_index]['Roll']

            speed_own = platforminfos_list[cur_index]['Speed']
            status_own = platforminfos_list[cur_index]['Status']
            leftweapon_own = platforminfos_list[cur_index]['LeftWeapon']

            own_list = [identification_own, type_own, availability_own, x_own, y_own,z_own, accmag_own,
                        heading_own, pitch_own, roll_own, speed_own, status_own, leftweapon_own]
            cur_agent_obs.extend(own_list)

            self.red_agent_loc[str(agent_order)] = {'X': x_own, 'Y': y_own, 'Z': z_own, 'heading': heading_own, 'pitch': pitch_own, 'roll': roll_own}

            team_list = []
            ## 拿其他队友的信息，对于某个Agent，它的队友固定为1，2，3，4顺序, 除了agent_order以外
            for i in range(5):
                if i != int(agent_order):
                    team_ID = RED_INFO[str(i)]['ID']
                    # 先查找这个队友还存在
                    team_exist = any(filter(lambda x: x['ID'] == team_ID, platforminfos_list))
                    if team_exist:
                        # 说明这个队友还活着，拿到这个队友的索引
                        team_index = next(((j, item) for j, item in enumerate(platforminfos_list) if item['ID'] == team_ID), (-1, None))[0]
                        identification_team = 0 if platforminfos_list[team_index]['Identification'] == '红方' else 1
                        type_team = platforminfos_list[team_index]['Type']
                        availability_team = platforminfos_list[team_index]['Availability']
                        x_team = platforminfos_list[team_index]['X']
                        y_team = platforminfos_list[team_index]['Y']
                        z_team = platforminfos_list[team_index]['Alt']
                        speed_team = platforminfos_list[team_index]['Speed']
                        team_ones = [identification_team, type_team, availability_team,
                                     x_team, y_team, z_team, speed_team]
                        team_list.extend(team_ones)
                    else:
                        # 说明这个队友已经死亡，那么只需要在对应位置上设为0即可
                        # 同时将这个队友的死亡ID记录
                        # print(f'死亡的队友有: {self.red_death}')
                        self.red_death.append(team_ID) if team_ID not in self.red_death else None
                        team_ones = [0., 0., 0., 0., 0., 0., 0.]
                        team_list.extend(team_ones)
            cur_agent_obs.extend(team_list)
            trackinfos_list = self.ori_message['red']['trackinfos']
            ## 拿敌人的状态信息
            enemy_list = []
            for i in range(5):
                enemy_ID = BLUE_INFO[str(i)]['ID']
                enemy_exist = any(filter(lambda x: x['ID'] == enemy_ID, trackinfos_list))
                if enemy_exist:
                    # 说明这个敌人还活着
                    enemy_index = next(((j, item) for j, item in enumerate(trackinfos_list) if item['ID'] == enemy_ID), (-1, None))[0]
                    # 找到这个敌人在trackinfos_list中的索引位置
                    identification_enemy = 0 if trackinfos_list[enemy_index]['Identification'] == '红方' else 1
                    type_enemy = trackinfos_list[enemy_index]['Type']
                    availability_enemy = trackinfos_list[enemy_index]['Availability']
                    x_enemy = trackinfos_list[enemy_index]['X']
                    y_enemy = trackinfos_list[enemy_index]['Y']
                    # z_enemy = self.llh_to_ecef(trackinfos_list[enemy_index]['Lon'], trackinfos_list[enemy_index]['Lat'],
                    #                  trackinfos_list[enemy_index]['Alt'])['Z']
                    z_enemy = trackinfos_list[enemy_index]['Alt']
                    speed_enemy = trackinfos_list[enemy_index]['Speed']
                    enemy_ones = [identification_enemy, type_enemy, availability_enemy,
                                 x_enemy, y_enemy, z_enemy, speed_enemy]
                    enemy_list.extend(enemy_ones)

                else:
                    # 说明这个敌人已经死掉了
                    self.blue_death.append(enemy_ID)
                    enemy_ones = [0., 0., 0., 0., 0., 0., 0.]
                    enemy_list.extend(enemy_ones)
            cur_agent_obs.extend(enemy_list)

            ## 拿敌人的导弹信息
            # 这里只取对自己有威胁的导弹
            missileinfos_list = self.ori_message['red']['missileinfos']
            cur_fire = []
            for i in range(12):
                # 一共是12枚导弹，
                # 分别查看这12枚导弹是否出现了
                cur_blue_fire_exist = any(filter(lambda x: x['Name'] == BLUE_FIRE_INFO[str(i)]['Name'], missileinfos_list))
                cur_blue_fire_index = next(((j, item) for j, item in enumerate(missileinfos_list) if item['Name'] == BLUE_FIRE_INFO[str(i)]['Name']), (-1, None))[0]
                # 如果出现了，查看是否锁定了自己？
                if cur_blue_fire_exist:
                    # 说明这枚弹已经出现了
                    # 拿到这枚弹的信息
                    cur_blue_fire_info = missileinfos_list[cur_blue_fire_index]
                    if cur_blue_fire_info['EngageTargetID'] == RED_INFO[str(agent_order)]['ID']:
                        # 锁定了自己
                        # cur_blue_fire_Z = self.llh_to_ecef(cur_blue_fire_info['Lon'], cur_blue_fire_info['Lat'],cur_blue_fire_info['Alt'])['Z']
                        cur_blue_fire_Z = cur_blue_fire_info['Alt']
                        cur_fire.extend([cur_blue_fire_info['X'],
                                         cur_blue_fire_info['Y'],
                                         cur_blue_fire_Z,
                                         cur_blue_fire_info['Speed']])
                    else:
                        # 未锁定自己
                        cur_fire.extend([0., 0., 0., 0.])
                else:
                    # 说明这枚弹没有出现
                    cur_fire.extend([0., 0., 0., 0.])

            cur_agent_obs.extend(cur_fire)

            # # 对每一个obs_value都进行归一化
            cur_agent_obs = cur_agent_obs - np.mean(cur_agent_obs)
            cur_agent_obs = cur_agent_obs / np.max(np.abs(cur_agent_obs))

        else:
            """当前Agent已经死亡"""
            cur_agent_obs.extend([0.0 for _ in range(124)])

        return cur_agent_obs

    def get_global_state(self):
        return np.array(self.obs).flatten()

    def get_state(self):
        """获取全局状态"""
        return self.get_global_state()

    def get_avail_actions(self):
        """
            获得可执行动作列表
            :return available_actions size = (5, 2, 5)
        """
        np1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=object)  # 9个方向
        np2 = np.array([1, 1, 1, 1, 1, 1], dtype=object)  # 5个攻击动作+1个不采用攻击动作
        avai_cow = np.array([np1, np2], dtype=object)
        avai_mask = np.tile(avai_cow, (5, 1))
        # 分别返回 agents的actions1，actions2
        avai_dict = {'0': [], '1': []}
        for i in range(5):
            """分别遍历五个Agent给出available mask"""
            # 先判断这个Agent是否已经死亡
            if_death = RED_INFO[str(i)]['ID'] in self.red_death  # 查看这个Agent的ID是否在死亡列表self.red_death中
            if if_death:
                avai_mask[i][0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=object)
                avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
            else:
                # 如果没有死亡，那么移动就可以全部是1,需要判断能不能攻击具体到某个敌方，要进行弹药数量判断和距离判断
                # cur_index = next(((j, item) for j, item in enumerate(self.ori_message['red']['platforminfos']) if RED_INFO[str(i)]['ID'] == self.ori_message['red']['platforminfos'][j])['ID'], (-1, None))[0]
                # next的正确用法。
                cur_index = next(((j, item) for j, item in enumerate(self.ori_message['red']['platforminfos']) if
                                  RED_INFO[str(i)]['ID'] == item.get('ID')), (-1, None))[0]

                left_weapon_num = self.ori_message['red']['platforminfos'][cur_index]['LeftWeapon']  # 先判断有没有剩余弹药
                cur_red_agent_loc = [self.ori_message['red']['platforminfos'][cur_index]['X'],
                                     self.ori_message['red']['platforminfos'][cur_index]['Y'],
                                     self.ori_message['red']['platforminfos'][cur_index]['Alt'],
                                     ]

                # print(f'agent ID: {RED_INFO[str(i)]["ID"]}, Leftweapon: {left_weapon_num}')
                if left_weapon_num == 0:
                    avai_mask[i][1] = np.array([0, 0, 0, 0, 0, 1], dtype=object)
                # else:
                elif left_weapon_num > 0:
                    # 进行地方距离判断
                    # avai_mask[i][1] 每个index上对应着敌方固定的实体
                    cur_ava = np.array([],dtype=object)
                    # 分别计算敌方实体的距离
                    for j in range(5):
                        # 查找蓝方实体的位置，要先判断这个蓝方是不是已经死掉了
                        blue_ID = BLUE_INFO[str(j)]['ID']
                        if blue_ID in self.blue_death:
                            cur_ava = np.append(cur_ava, 0)
                        else:
                            # 这个蓝方实体还没有死掉
                            enemy_index = next(((i_, item) for i_, item in enumerate(self.ori_message['red']['trackinfos']) if
                                 item.get('ID') == blue_ID), (-1, None))[0]
                            enemy_infos = self.ori_message['red']['trackinfos'][enemy_index]
                            enemy_loc = [
                                enemy_infos['X'],
                                enemy_infos['Y'],
                                enemy_infos['Alt']
                            ]
                            # 判断距离
                            if self.distance_computation(cur_red_agent_loc, enemy_loc) <= 80000:
                                cur_ava = np.append(cur_ava, 1)
                            else:
                                cur_ava = np.append(cur_ava, 0)

                    # 加入射击间隔
                    agent_shoot_interval = self.shoot_interval[i]
                    agent_shoot_legal = np.ones_like(agent_shoot_interval)
                    agent_shoot_legal[agent_shoot_interval != 0] = 0
                    # print(f'agent_shoot_legal', agent_shoot_legal)
                    # print(f'original cur_ava: {cur_ava}')
                    new_cur_ava = np.zeros_like(cur_ava)
                    new_cur_ava[(cur_ava == 1) & (agent_shoot_legal == 1)] = 1
                    # print(f'cur_ava', new_cur_ava)
                    # cur_ava.append(1)  # 最后一维是不执行任何攻击工作
                    cur_ava = np.append(new_cur_ava, 1)
                    # cur_ava = np.append(cur_ava, 1)
                    avai_mask[i][1] = cur_ava

        for i in range(5):
            avai_dict['0'].append(list(avai_mask[i][0]))
            avai_dict['1'].append(list(avai_mask[i][1]))

        return avai_dict

    def distance_computation(self, point_1=None, point_2=None):
        """
            计算两个点之间的距离
            :param point_1 第一个点的 [X，Y，Z]
            :param point_2 第二个点的 [X，Y，Z]

            :return distance: int
        """
        return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2 + (point_1[2] - point_2[2])**2)

    def close_environ(self):
        """关闭仿真引擎"""
        self._end()

    def render(self):
        """渲染视频"""
        pass

    def save_replay(self):
        pass

    def llh_to_ecef(self, lat, lon, alt) -> {}:
        # WGS84椭球参数
        a = 6378137.0  # 地球半径
        e = 8.1819190842622e-2  # 偏心率
        # 将度转换为弧度
        # lat = math.radians(lat)
        # lon = math.radians(lon)
        N = a / math.sqrt(1 - e ** 2 * math.sin(lat) ** 2)
        X = (N + alt) * math.cos(lat) * math.cos(lon)
        Y = (N + alt) * math.cos(lat) * math.sin(lon)
        Z = ((1 - e ** 2) * N + alt) * math.sin(lat)

        return {'X': X, 'Y': Y, 'Z': Z}

    def get_total_actions(self):
        return self.action_space_1 + self.action_space_2

    def get_obs_size(self):
        return self.observation_space # 124

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(), # 620
            "obs_shape": self.get_obs_size(), # 124
            # "n_actions": self.get_total_actions(), # 9 + 6 = 15
            "n_actions_1": self.action_space_1, # 9
            "n_actions_2": self.action_space_2, # 6
            "n_agents": self.n_agents, # 5
            "episode_limit": self.episode_limit
        }

        return env_info

    def render(self):
        pass

    def get_stats(self):
        stats = {
            "battles_won": self.battles_won,
            "battles_game": self.battles_game,
            "win_rate": self.battles_won / self.battles_game
        }
        return stats


if __name__ == '__main__':
    """测试环境类"""
    print('testing environment')
    address = ADDRESS['ip'] + ":" + str(ADDRESS['port'])
    print('address:', address)
    test_env = HuaRuBattleEnvWrapper(config['agents'], address)
    test_env.reset()
    # test_env.get_available_actions()
