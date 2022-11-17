# 修改状态空间，改为障碍的观测和目标点的位置差（仅1个）

# 2022.4.28修正版，此种方法由无人机主动探索

import sys, os

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path

import datetime
import torch
import random
from env import NormalizedActions, OUNoise
from agent import DDPG
from utils import save_results, make_dir
from plot import plot_rewards, plot_rewards_cn
from Method import getReward, setup_seed
from ApfAlgorithm import APF
import numpy as np
import pandas as pd

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'  # 算法名称
        self.env = 'UAV-Task4'  # 环境名称
        self.result_path = curr_path + "/outputs/" + self.env + \
                           '/' + curr_time + '/results/'  # 保存结果的路径
        self.model_path = curr_path + "/outputs/" + self.env + \
                          '/' + curr_time + '/models/'  # 保存模型的路径
        self.data_path = curr_path + "/outputs/" + self.env + \
                         '/' + curr_time + '/data_csv/'  # 存储数据的路径
        self.train_eps = 1000  # 训练的回合数
        self.eval_eps = 10  # 测试的回合数
        self.gamma = 0.99  # 折扣因子
        self.critic_lr = 1e-3  # 评论家网络的学习率
        self.actor_lr = 1e-4  # 演员网络的学习率
        self.memory_capacity = 8000
        self.batch_size = 128
        self.target_update = 2
        self.hidden_dim = 256
        self.soft_tau = 1e-2  # 软更新参数
        self.max_step = 500
        self.update_every = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_list = []
        self.flag_list = []
        self.eval_action = []
        self.collision_count_list = []


def env_agent_config(cfg, seed):
    env = APF()
    setup_seed(seed)  # 随机种子
    state_dim = 3 * (env.numberOfSphere + env.numberOfCylinder + env.numberOfCone) + 3  # 障碍物的三维和目标点的三维

    # # task用这个动作维度
    # action_dim = 1 * (env.numberOfSphere + env.numberOfCylinder + env.numberOfCone)

    # task2用这个动作维度
    # 记得修改env文件中的动作维度
    # 记得修改ApfAlgorithm中的action_bound
    action_dim = 3
    agent = DDPG(state_dim, action_dim, cfg)
    env.get_time(cfg.data_path)
    return env, agent


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    ou_noise = OUNoise(env)  # 动作噪声
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    collision_count_list = []
    MAX_REWARD = float('-inf')
    for i_ep in range(cfg.train_eps):
        q = env.x0
        action_list = []
        flag_list = []
        env.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        q_before = [None, None, None]
        collision_count = 0
        for i_step in range(cfg.max_step):
            i_step += 1
            collision_flag = 0
            temp_index = 0
            obsDicq = env.calculateDynamicState2(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            sAll = env.qgoal - q
            sAll = sAll.tolist()
            obs_mix = obs_sphere + obs_cylinder + obs_cone + sAll
            obs = np.array([])  # 中心控制器接受所有状态集合
            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k]))  # 拼接状态为一个1*n向量
            # obs = np.hstack((obs, q))
            if i_ep > 100:
                action = agent.choose_action(obs, "xunlian")  # 在执行后再正则化
                action = ou_noise.get_action(action, i_step)
                # action_sphere = action[0:env.numberOfSphere]
                # action_cylinder = action[env.numberOfSphere:env.numberOfSphere + env.numberOfCylinder]
                # action_cone = action[env.numberOfSphere + env.numberOfCylinder:env.numberOfSphere + \
                #                                                                env.numberOfCylinder + env.numberOfCone]
            else:
                # action_sphere = [random.uniform(env.act_bound[0], env.act_bound[1]) for k in range(env.numberOfSphere)]
                # action_cylinder = [random.uniform(env.act_bound[0], env.act_bound[1]) for k in
                #                    range(env.numberOfCylinder)]
                # action_cone = [random.uniform(env.act_bound[0], env.act_bound[1]) for k in range(env.numberOfCone)]
                action_x = random.uniform(env.act_bound[0], env.act_bound[1])
                action_y = random.uniform(env.act_bound[0], env.act_bound[1])
                action_z = random.uniform(env.act_bound[0], env.act_bound[1])
                action = np.array([action_x, action_y, action_z])
            # q_next = env.getqNext(env.epsilon0, action_sphere, action_cylinder, action_cone, q, q_before)

            # q_next = env.getqNext2(env.epsilon0, action, q, q_before)
            # flag = env.checkCollision(q_next)
            if i_ep > 100:
                xunhuan_count = 0
                while collision_flag != 1:
                    xunhuan_count += 1
                    q_next = env.getqNext2(env.epsilon0, action.copy(), q.copy(), q_before.copy())
                    flag = env.checkCollision(q_next)
                    if xunhuan_count > 15:
                        break
                    if (flag == np.array([1, -1, -1])).all():
                        collision_flag = 1  # 没碰撞
                    else:
                        temp_index = 1
                        collision_index = env.checkAction(q_next.copy(), flag.copy(), action.copy(), q.copy())
                        if collision_index[0] == 1:
                            if 0 < action[0] <= 1:
                                action[0] -= 0.1
                                action[2] += 0.1
                            elif -1 <= action[0] < 0:
                                action[0] += 0.1
                                action[2] += 0.1
                        if collision_index[1] == 1:
                            if 0 < action[1] <= 1:
                                action[1] -= 0.1
                                action[2] += 0.1
                            elif -1 <= action[1] < 0:
                                action[1] += 0.1
                                action[2] += 0.1
                        if collision_index[2] == 1:
                            if 0 < action[2] <= 1:
                                action[2] -= 0.1
                            elif -1 <= action[2] < 0:
                                action[2] += 0.1
            action_list.append(action)
            # if temp_index == 0:
            #     collision_count += 0
            # elif temp_index == 1:
            #     collision_count += 1
            q_next = env.getqNext2(env.epsilon0, action, q, q_before)
            env.path = np.vstack((env.path, q_next))
            flag = env.checkCollision(q_next)
            if (flag == np.array([1, -1, -1])).all():
                collision_count += 0
            else:
                collision_count += 1
            flag_list.append(flag)
            obsDicqNext = env.calculateDynamicState2(q_next)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], \
                                                                obsDicqNext['cone']
            sAll_next = env.qgoal - q_next
            sAll_next = sAll_next.tolist()
            obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next + sAll_next
            obs_next = np.array([])
            for k in range(len(obs_mix_next)):
                obs_next = np.hstack((obs_next, obs_mix_next[k]))
            # obs_next = np.hstack((obs_next, q_next))
            reward = getReward(flag, env, q_before, q, q_next)
            ep_reward += reward
            done = True if env.distanceCost(env.qgoal, q_next) < env.threshold else False
            # centralizedContriller.replay_buffer.store(obs, action, reward, obs_next, done)
            # next_state, reward, done, _ = env.step(action)  # 在step函数中正则化
            agent.memory.push(obs, action, reward, obs_next, done)
            if i_ep >= 100 and i_step % cfg.update_every == 0:
                agent.update()
            if done:
                break
            q_before = q.copy()
            q = q_next.copy()
        if (i_ep + 1) % 10 == 0:
            print('回合：{}/{}，奖励：{:.2f}，碰撞次数：{:.2f}'.format(i_ep + 1, cfg.train_eps, ep_reward, collision_count))
        rewards.append(ep_reward)
        cfg.action_list.append(action_list)
        cfg.flag_list.append(flag_list)

        if collision_count_list:
            collision_count_list.append(0.9 * collision_count_list[-1] + 0.1 * collision_count)
        else:
            collision_count_list.append(collision_count)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        cfg.collision_count_list.append(collision_count_list)
        # if ep_reward > MAX_REWARD and ep_reward > -50 and i_ep > 100:
        #     MAX_REWARD = ep_reward
        #     agent.save_mid(cfg.model_path)
    print('完成训练！')
    return rewards, ma_rewards, collision_count_list


def eval(cfg, env, agent, jieduan):
    print('开始测试！')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(cfg.eval_eps):
        q = env.x0
        env.reset()
        done = False
        ep_reward = 0
        i_step = 0
        q_before = [None, None, None]
        collision_count = 0
        action_list = []
        for j in range(cfg.max_step):
            i_step += 1
            collision_flag = 0
            obsDicq = env.calculateDynamicState2(q)
            obs_sphere, obs_cylinder, obs_cone = obsDicq['sphere'], obsDicq['cylinder'], obsDicq['cone']
            sAll = env.qgoal - q
            sAll = sAll.tolist()
            obs_mix = obs_sphere + obs_cylinder + obs_cone + sAll
            obs = np.array([])  # 中心控制器接受所有状态集合
            for k in range(len(obs_mix)):
                obs = np.hstack((obs, obs_mix[k]))  # 拼接状态为一个1*n向量
            # obs = np.hstack((obs, q))
            action = agent.choose_action(obs, "ceshi")
            action = action.reshape(-1)
            action = action.cpu()
            action = action.detach().numpy()
            xunhuan_count = 0
            while collision_flag != 1:
                xunhuan_count += 1
                q_next = env.getqNext2(env.epsilon0, action.copy(), q.copy(), q_before.copy())
                flag = env.checkCollision(q_next)
                if xunhuan_count > 15:
                    break
                if (flag == np.array([1, -1, -1])).all():
                    collision_flag = 1  # 没碰撞
                else:
                    temp_index = 1
                    collision_index = env.checkAction(q_next.copy(), flag.copy(), action.copy(), q.copy())
                    if collision_index[0] == 1:
                        if 0 < action[0] <= 1:
                            action[0] -= 0.1
                            action[2] += 0.1
                        elif -1 <= action[0] < 0:
                            action[0] += 0.1
                            action[2] += 0.1
                    if collision_index[1] == 1:
                        if 0 < action[1] <= 1:
                            action[1] -= 0.1
                            action[2] += 0.1
                        elif -1 <= action[1] < 0:
                            action[1] += 0.1
                            action[2] += 0.1
                    if collision_index[2] == 1:
                        if 0 < action[2] <= 1:
                            action[2] -= 0.1
                        elif -1 <= action[2] < 0:
                            action[2] += 0.1
            action_list.append(action)
            # action = action.reshape(-1)
            # action = action.tolist()
            # action_sphere = action[0:env.numberOfSphere]
            # action_cylinder = action[env.numberOfSphere:env.numberOfSphere + env.numberOfCylinder]
            # action_cone = action[env.numberOfSphere + env.numberOfCylinder:env.numberOfSphere + \
            #                                                                env.numberOfCylinder + env.numberOfCone]

            q_next = env.getqNext2(env.epsilon0, action, q, q_before)
            env.path = np.vstack((env.path, q_next))
            obsDicqNext = env.calculateDynamicState2(q_next)
            obs_sphere_next, obs_cylinder_next, obs_cone_next = obsDicqNext['sphere'], obsDicqNext['cylinder'], \
                                                                obsDicqNext['cone']
            sAll_next = env.qgoal - q_next
            sAll_next = sAll_next.tolist()
            obs_mix_next = obs_sphere_next + obs_cylinder_next + obs_cone_next + sAll_next
            obs_next = np.array([])
            for k in range(len(obs_mix_next)):
                obs_next = np.hstack((obs_next, obs_mix_next[k]))
            # obs_next = np.hstack((obs_next, q_next))
            flag = env.checkCollision(q_next)
            if (flag == np.array([1, -1, -1])).all():
                collision_count += 0
            else:
                collision_count += 1
            reward = getReward(flag, env, q_before, q, q_next)
            ep_reward += reward
            done = True if env.distanceCost(env.qgoal, q_next) < env.threshold else False
            if done:
                break
            q_before = q.copy()
            q = q_next.copy()
        print('回合：{}/{}, 奖励：{},碰撞次数：{:.2f}'.format(i_ep + 1, cfg.eval_eps, ep_reward, collision_count))
        cfg.eval_action.append(action_list)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    if jieduan == 'final':
        env.drawPath(cfg.result_path)
    elif jieduan == 'mid':
        env.drawPathmid(cfg.result_path)
    print('完成测试！')
    env.saveCSV()
    cfg.eval_action = np.array(cfg.eval_action)
    return rewards, ma_rewards


if __name__ == '__main__':
    cfg = DDPGConfig()
    env, agent = env_agent_config(cfg, seed=4)
    make_dir(cfg.result_path, cfg.model_path, cfg.data_path)
    start = datetime.datetime.now()
    rewards, ma_rewards, count_list = train(cfg, env, agent)
    end = datetime.datetime.now()
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train_task4_' + env.env_name, path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="train_task4_" + env.env_name, env=cfg.env, algo=cfg.algo,
                    path=cfg.result_path)
    writer = pd.ExcelWriter(r'D:\LY_UAV\UAV_task4_o4.xlsx')
    df1 = pd.DataFrame(ma_rewards, columns=['ma_rewards'])
    df2 = pd.DataFrame(count_list, columns=['count_list'])
    df1.to_excel(writer, sheet_name='df1')
    df2.to_excel(writer, sheet_name='df2')
    writer.close()
    # 测试
    env, agent = env_agent_config(cfg, seed=4)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent, 'final')
    save_results(rewards, ma_rewards, tag='eval_task4_' + env.env_name, path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="eval_task4_" + env.env_name, env=cfg.env, algo=cfg.algo,
                    path=cfg.result_path)
    print(end - start)

    # agent.load_mid(path=cfg.model_path)
    # rewards, ma_rewards = eval(cfg, env, agent, 'mid')
    # save_results(rewards, ma_rewards, tag='mid_eval_task2', path=cfg.result_path)
    # plot_rewards_cn(rewards, ma_rewards, tag="mid_eval_task2", env=cfg.env, algo=cfg.algo, path=cfg.result_path)
