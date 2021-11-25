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


def run_experiment(json_config, eval: bool):
    json_config["args"]["saved_episode"] = json_config["args"]["max_episode"] - 1  # TODO adjust this in exp.conf
    json_config["args"]["load_episode_saved"] = json_config["args"]["max_episode"] - 1  # TODO adjust this in exp.conf
    arglist = generate_arglist(json_config["args"])
    arglist.eval = eval
    env_config = json_config["env_config"]
    hash_ = f"{arglist.scenario}/{json_config['hash']}"
    if eval:
        os.makedirs(f'experiments/{hash_}', exist_ok=True)
        with open(f'experiments/{hash_}/config.json', "w") as f:
            json.dump(json_config, f)
        df = experiment_execution(arglist, env_config, root="experiments", hash=hash_)
        df.to_csv(f"experiments/{hash_}/data.csv")
    else:
        experiment_execution(arglist, env_config, root="experiments", hash=hash_)


def experiment_execution(arglist, env_config, root="experiments", hash="default", verbose=False):
    logger = logging.getLogger(f'{root}_{hash}')
    if arglist.eval:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{root}/{hash}/log.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    print(hash)
    ACTORS = 1
    env = EnvWrapper(arglist.scenario, ACTORS, arglist.saved_episode, configuration=env_config)

    df = pd.DataFrame()

    maddpg_wrapper = MADDPG(ACTORS, hash, arglist)
    maddpg_wrapper.create_agents(env, arglist, root=root, hash=hash)
    epsilon = arglist.epsilon
    j = 0
    for episode in range(0, arglist.max_episode):
        obs = env.reset()
        maddpg_wrapper.reset()
        total_reward = [[] for i in maddpg_wrapper.workers]
        step = 0
        reward = None
        logger.info(f"Episode: {episode}")
        all_diff = []
        while step < 100:
            if not arglist.eval:
                env.render(0, title=hash)
                time.sleep(0.03)

            actions = maddpg_wrapper.take_actions(obs, at_random=arglist.eval, eps=epsilon if arglist.eval else 0.0)
            obs2, reward, done, _ = env.step(actions)

            for actor in range(ACTORS):
                for i, rew in enumerate(reward[actor]):
                    total_reward[i].append(rew)

            j += ACTORS
            # terminal = all(done)
            if arglist.eval:
                start = time.time()
                maddpg_wrapper.update(j, ACTORS, actions, reward, obs, obs2, done)
                diff_ = time.time() - start
                all_diff.append(diff_)

            obs = obs2
            step += 1
        if reward:
            print(list(map(np.mean, total_reward)))
            logger.info(f"episode: {episode}, mean reward: {list(map(np.mean, total_reward))}")

        episode_data = {f"agent{i}_reward": rew for i, rew in enumerate(list(map(np.mean, total_reward)))}
        episode_data["mean_diff"] = np.mean(all_diff)
        for worker, ep_ave_max in zip(maddpg_wrapper.workers, maddpg_wrapper.ep_ave_max_q_value):
            episode_data[f"agent{worker.pos}_max_q_val"] = ep_ave_max
        episode_data["episode"] = episode
        df = df.append(episode_data, ignore_index=True)

        if arglist.eval and episode % arglist.saved_episode == 0 and episode > 0:
            maddpg_wrapper.save(episode, root=root, logger=logger)

        if arglist.eval and episode % 100 == 0 and epsilon > arglist.epsilon_min:
            epsilon *= arglist.epsilon_decay

    df.to_csv(f"{root}/{hash}/data.csv")
    return df


def run_by_hash(hash: str, eval: bool):
    with open("exp_config.json", "r") as f:
        configurations = json.load(f)
    for json_config in configurations:
        if json_config["hash"] == hash:
            run_experiment(json_config, eval)
            return
    raise Exception(f"Could not find hash: {hash}")


def run_by_idx(exp_idx: int, eval: bool):
    with open("exp_config.json", "r") as f:
        configurations = json.load(f)

    run_experiment(configurations[exp_idx], eval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experiments for Reinforcement learning')
    parser.add_argument("--hash", type=str, default=None)
    parser.add_argument("--exp", type=int, default=None)
    parser.add_argument("--eval", type=bool, default=True)
    arglist = parser.parse_args()
    if arglist.hash is not None:
        run_by_hash(arglist.hash, arglist.eval)
    elif arglist.exp is not None:
        run_by_idx(arglist.exp, arglist.eval)
    else:
        raise Exception("Please specify either --hash or --idx in order to select an experiment")
