import json
import logging
from torch.multiprocessing import Pool

import gym
import time
import numpy as np
import pandas as pd
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


def generate_arglist(config_arglist):
    namespace = argparse.Namespace()
    for k, v in config_arglist.items():
        if not hasattr(namespace, k):
            setattr(namespace, k, v)
    return namespace


def run_experiment(json_config, benchmark: bool, benchmark_episodes: int):
    json_config["args"]["saved_episode"] = json_config["args"]["max_episode"] - 1
    json_config["args"]["load_episode_saved"] = json_config["args"]["max_episode"] - 1
    arglist = generate_arglist(json_config["args"])
    arglist.benchmark = benchmark
    env_config = json_config["env_config"]
    hash_ = f"{arglist.scenario}/{json_config['hash']}"
    if benchmark:
        os.makedirs(f'experiments/{hash_}/benchmark', exist_ok=True)
        with open(f'experiments/{hash_}/config.json', "w") as f:
            json.dump(json_config, f)
        experiment_execution(arglist, env_config, root="experiments", hash=hash_, benchmark_episodes=benchmark_episodes)
    else:
        pass


def experiment_execution(arglist, env_config, root="experiments", hash="default", benchmark_episodes=100,
                         verbose=False):
    print(hash)

    ACTORS = 1
    env = EnvWrapper(arglist.scenario, ACTORS, arglist.saved_episode, configuration=env_config,
                     benchmark=arglist.benchmark)
    rew_array = np.zeros((len(env.get_env().agents), arglist.max_episode))
    maddpg_wrapper = MADDPG(ACTORS, arglist.scenario, arglist)

    maddpg_wrapper.create_agents(env, arglist, root=root, hash=hash)
    epsilon = arglist.epsilon
    j = 0
    final_bench_list = []
    for episode in range(benchmark_episodes):
        obs = env.reset()
        maddpg_wrapper.reset()
        total_reward = [[] for i in maddpg_wrapper.workers]
        step = 0
        max_steps = 100
        reward = None
        bench_list = []
        while step < max_steps:
            # if arglist.benchmark:
            # env.render(0)
            # time.sleep(0.1)

            actions = maddpg_wrapper.take_actions(obs,
                                                  at_random=False,
                                                  eps=epsilon)
            obs2, reward, done, info = env.step(actions)

            for actor in range(ACTORS):
                for i, rew in enumerate(reward[actor]):
                    total_reward[i].append(rew)

            j += ACTORS
            obs = obs2
            step += 1

            dict_list = info[0]['n']
            if step == 1:  # copy the list in the first step
                bench_list = dict_list
            else:  # accumulate values for each key for all steps of the episode
                for i, dict in enumerate(dict_list):
                    for key in dict:
                        if key == "still_active_foods":  # this is for tragedy of commons
                            bench_list[i][key] = bench_list[i][key] or dict_list[i][key]
                        else:
                            bench_list[i][key] += dict_list[i][key]

        if episode == 0:  # copy the list
            final_bench_list = bench_list
            for i, dict in enumerate(bench_list):
                final_bench_list[i]['reward'] = sum(total_reward[i])
                for key in dict:  # convert integer to list element, to append later
                    final_bench_list[i][key] = [final_bench_list[i][key]]
        else:
            for i, dict in enumerate(bench_list):
                final_bench_list[i]['reward'].append(sum(total_reward[i]))
                for key in dict:  # append to key of dict for each agent
                    final_bench_list[i][key].append(bench_list[i][key])

        rew_array[:, episode] = np.asarray(list(map(np.mean, total_reward)))

    if arglist.benchmark:
        for i in range(len(final_bench_list)):
            df = pd.DataFrame.from_dict(final_bench_list[i])
            df.to_csv(f"{root}/{hash}/benchmark/bm_agent" + str(i) + ".csv")
    return


def run_by_hash(hash: str, benchmark: bool, benchmark_episodes: int):
    with open("exp_config.json", "r") as f:
        configurations = json.load(f)
    for json_config in configurations:
        if json_config["hash"] == hash:
            run_experiment(json_config, benchmark, benchmark_episodes)
            return
    raise Exception(f"Could not find hash: {hash}")


def run_by_idx(exp_idx: int, benchmark: bool, benchmark_episodes: int):
    with open("exp_config.json", "r") as f:
        configurations = json.load(f)
    run_experiment(configurations[exp_idx], benchmark, benchmark_episodes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Benchmarks for Reinforcement learning')
    parser.add_argument("--hash", type=str, default=None)
    parser.add_argument("--exp", type=int, default=None)
    parser.add_argument("--eval", type=bool, default=False)
    parser.add_argument("--benchmark", type=bool, default=True)
    parser.add_argument("--benchmark_episodes", type=int, default=50)
    arglist = parser.parse_args()
    if arglist.hash is not None:
        run_by_hash(arglist.hash, arglist.benchmark, arglist.benchmark_episodes)
    elif arglist.exp is not None:
        run_by_idx(arglist.exp, arglist.benchmark, arglist.benchmark_episodes)
    else:
        raise Exception("Please specify either --hash or --idx in order to select an experiment")
