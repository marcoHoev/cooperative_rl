import json
import random

import gym
import time
import numpy as np
import ddpg
import os
import agent
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Box
from maddpg import MADDPG
from time import gmtime, strftime
from env_wrapper import EnvWrapper
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def parse_args():
    parser = argparse.ArgumentParser('Reinforcement Learning parser for DDPG')
    parser.add_argument('--scenario', type=str, default='hunter_gatherer_teams')
    parser.add_argument('--eval', action='store_false')
    parser.add_argument('--load-episode-saved', type=int, default=80000)
    parser.add_argument('--saved-episode', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--max-episode', type=int, default=80001)  # 100000
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--tau', type=float, default=0.01)
    # Epsilon for exploration
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--epsilon-decay', type=float, default=0.9)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--plot-mode', type=str, choices=['loop', 'save'], default='save')
    return parser.parse_args()


def create_plot(rew_array, episode, mode, scenario):
    plt.clf()
    for row in range(rew_array.shape[0]):
        plt.plot(np.arange(episode), rew_array[row, 0:episode], label="Agent %d" % row)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    if mode == 'loop':
        plt.pause(0.0001)
    elif mode == 'save':
        print(f"saved/{scenario}/plot.png")
        plt.savefig(f"saved/{scenario}/plot.png")  # for report, change this to pdf or create df
    else:
        plt.show()


def main(arglist, env_config=None):
    ACTORS = 1
    env = EnvWrapper(arglist.scenario, ACTORS, arglist.saved_episode, configuration=env_config)
    rew_array = np.zeros((len(env.get_env().agents), arglist.max_episode))
    if arglist.eval:
        current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        writer = SummaryWriter(log_dir='./logs/' + current_time + '-' + arglist.scenario)
    maddpg_wrapper = MADDPG(ACTORS, arglist.scenario, arglist)

    maddpg_wrapper.create_agents(env, arglist)
    epsilon = arglist.epsilon
    j = 0
    for episode in range(arglist.max_episode):
        obs = env.reset()
        maddpg_wrapper.reset()
        total_reward = [[] for i in maddpg_wrapper.workers]
        step = 0
        reward = None
        while step < 100:
            if not arglist.eval:
                env.render(0)
                time.sleep(0.03)

            actions = maddpg_wrapper.take_actions(obs,
                                                  at_random=arglist.eval,
                                                  eps=epsilon)
            obs2, reward, done, _ = env.step(actions)

            for actor in range(ACTORS):
                for i, rew in enumerate(reward[actor]):
                    total_reward[i].append(rew)

            j += ACTORS
            # terminal = all(done)
            if arglist.eval:
                maddpg_wrapper.update(j, ACTORS, actions, reward, obs, obs2, done)

            obs = obs2
            step += 1
        if reward:
            print(reward)
        if arglist.eval:
            for worker, ep_ave_max in zip(maddpg_wrapper.workers, maddpg_wrapper.ep_ave_max_q_value):
                print(worker.pos, ' => average_max_q: ', f"{ep_ave_max / float(step):.6f}", ' Reward: ',
                      f"{np.mean(total_reward[worker.pos]):.6f}", ' Episode: ', episode, f"Epsilon: {epsilon:.6f}")
                writer.add_scalar(str(worker.pos) + '/Average_max_q', ep_ave_max / float(step), episode)
                writer.add_scalar(str(worker.pos) + '/Reward Agent', total_reward[worker.pos][-1], episode)

        if arglist.eval and episode % 100 == 0 and epsilon > arglist.epsilon_min:
            epsilon *= arglist.epsilon_decay
        rew_array[:, episode] = np.asarray(list(map(np.mean, total_reward)))

        if arglist.eval and episode % arglist.saved_episode == 0 and episode > 0:
            maddpg_wrapper.save(episode)

        if arglist.plot_mode == "loop":
            if episode % arglist.saved_episode == 0 and episode > 0:
                create_plot(rew_array, episode, 'loop', arglist.scenario)
        elif arglist.plot_mode == "save":
            if episode == arglist.max_episode - 1:
                create_plot(rew_array, episode, 'save', arglist.scenario)
        else:
            print(f"Warning non supported plot_mode: {arglist.plot_mode}")


if __name__ == "__main__":
    arglist = parse_args()
    print(arglist)
    try:
        os.mkdir(f'./saved/{arglist.scenario}')
    except OSError:
        print("Creation of the directory failed. If the folder exists, content will be overwritten")
    else:
        print("Successfully created the directory")
    main(arglist)
