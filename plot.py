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
import seaborn as sns
from auto_benchmark import run_by_hash as generate_data_by_hash

COMPARISON_EPISODES = 1000


def get_configs_by_hash(hashes: [str]):
    with open("exp_config.json", "r") as f:
        configurations = json.load(f)
    exp_configs = []
    for json_config in configurations:

        for hash in hashes:
            if json_config["hash"] == hash:
                exp_configs.append(json_config)
    return exp_configs


def idx_to_agent_lookup(idx, nr_agents):
    if nr_agents == 4:
        tmp = {0: "red_hunter_0", 1: "blue_hunter_0", 2: "blue_gatherer_0", 3: "red_gatherer_0"}
    else:  # we have min 4 max 8 agents
        tmp = {0: "red_hunter_0", 1: "red_hunter_1",
               2: "blue_hunter_0", 3: "blue_hunter_1",
               4: "blue_gatherer_0", 5: "blue_gatherer_1",
               6: "red_gatherer_0", 7: "red_gatherer_1"}
    return tmp[idx]


def merge_agent_dfs(config, nr_agents, root="experiments", is_hunter=True):
    path = f"{root}/{config['args']['scenario']}/{config['hash']}/benchmark"
    df_all = pd.DataFrame()
    for idx in range(nr_agents):
        df = pd.read_csv(f"{path}/bm_agent{idx}.csv")
        if is_hunter:
            df = df.assign(agent_idx=idx_to_agent_lookup(idx, nr_agents))
        else:
            df = df.assign(agent_idx=idx)

        del df["Unnamed: 0"]
        df_all = df_all.append(df)
    return df_all


def get_nr_agents_tragedy_commons(config):
    return config["env_config"]["AGENTS_NR"]


def get_nr_agents_hunter_gatherer(config):
    env_cfg = config["env_config"]
    return env_cfg["GATHERERS_RED"] + env_cfg["HUNTERS_RED"] + \
           env_cfg["GATHERERS_BLUE"] + env_cfg["HUNTERS_BLUE"]


def generate_all_data(hashes):
    for hash in hashes:
        print(f"Evaluating {hash}...")
        generate_data_by_hash(hash, True, COMPARISON_EPISODES)


def plot_hashes(hashes, name, comparison_column, save=False, is_hunter=True):
    configs = get_configs_by_hash(hashes)
    df_all = pd.DataFrame()
    for config in configs:
        nr_agents = get_nr_agents_hunter_gatherer(config) if is_hunter else get_nr_agents_tragedy_commons(config)
        df = merge_agent_dfs(config, nr_agents, is_hunter=is_hunter)
        df = df.assign(hash=config["hash"])

        df_all = df_all.append(df)
    if is_hunter:
        palette = {"red_hunter_0": "tomato", "red_hunter_1": "tomato",
                   "blue_hunter_0": "darkslateblue", "blue_hunter_1": "darkslateblue",
                   "blue_gatherer_0": "slateblue", "blue_gatherer_1": "slateblue",
                   "red_gatherer_0": "darksalmon", "red_gatherer_1": "darksalmon"}
        sns.barplot(x="hash", hue="agent_idx", y=comparison_column, data=df_all, errwidth=1, capsize=0.1,
                    palette=palette)
    else:
        palette = {0: "tomato", 1: "darkslateblue", 2: "slateblue", 3: "darksalmon"}
        if comparison_column == "still_active_foods":
            df_all = df_all[df_all["agent_idx"] == 0]
            df_all = df_all.groupby("hash").sum().reset_index()
            df_all["still_active_foods"] = df_all["still_active_foods"] / (COMPARISON_EPISODES)
            sns.barplot(x="hash", y=comparison_column, data=df_all, errwidth=1, capsize=0.1)
        else:
            sns.barplot(x="hash", hue="agent_idx", y=comparison_column, data=df_all, errwidth=1, capsize=0.1)

    plt.title(f"{name}")
    if save:
        save_dir = f"plots/{'hunter_gatherer_teams' if is_hunter else 'tragedy_commons'}"
        os.makedirs(f"plots/{'hunter_gatherer_teams' if is_hunter else 'tragedy_commons'}", exist_ok=True)
        plt.savefig(f"{save_dir}/{name}.pdf")
        plt.clf()
    else:
        plt.show()


def plot_tragedy_commons(generate_data=False, save=True):
    if generate_data:
        all_hashes = [
            "desert_0_ddpg", "desert_250_ddpg", "desert_500_ddpg",
            "desert_0_maddpg", "desert_250_maddpg", "desert_500_maddpg",
            "nrfood_2_ddpg", "nrfood_3_ddpg", "nrfood_4_ddpg",
            "nrfood_2_maddpg", "nrfood_3_maddpg", "nrfood_4_maddpg",
            "prew_10_ddpg", "prew_10_maddpg",
            "prew_50_ddpg", "prew_50_maddpg",
            "prew_90_ddpg", "prew_90_maddpg"
        ]
        generate_all_data(all_hashes)
    plot_hashes(["prew_10_ddpg", "prew_50_ddpg", "prew_90_ddpg", ],
                "Collection of foods ddpg",
                "agent_collected",
                save=save,
                is_hunter=False)
    plot_hashes(["prew_10_ddpg", "prew_50_ddpg", "prew_90_ddpg", ],
                "Foods brought home ddpg",
                "agent_collected_home",
                save=save,
                is_hunter=False)

    plot_hashes(["prew_10_maddpg", "prew_50_maddpg", "prew_90_maddpg", ],
                "Collection of foods maddpg",
                "agent_collected",
                save=save,
                is_hunter=False)
    plot_hashes(["prew_10_maddpg", "prew_50_maddpg", "prew_90_maddpg", ],
                "Foods brought home maddpg",
                "agent_collected_home",
                save=save,
                is_hunter=False)

    plot_hashes(["desert_0_maddpg", "desert_250_maddpg", "desert_500_maddpg", ],
                "Collection of food orbs MADDPG",
                "agent_collected",
                save=save,
                is_hunter=False)

    plot_hashes(["desert_0_ddpg", "desert_250_ddpg", "desert_500_ddpg", ],
                "Collection of food orbs DDPG",
                "agent_collected",
                save=save,
                is_hunter=False)

    plot_hashes(["desert_0_maddpg", "desert_250_maddpg", "desert_500_maddpg", ],
                "Reward of desert scenarios MADDPG",
                "reward",
                save=save,
                is_hunter=False)

    plot_hashes(["desert_0_ddpg", "desert_250_ddpg", "desert_500_ddpg", ],
                "Reward of desert scenarious DDPG",
                "reward",
                save=save,
                is_hunter=False)


def plot_hunter_gatherer(generate_data=True, save=False):
    if generate_data:
        all_hashes = ["base_maddpg", "ddpg"]
        generate_all_data(all_hashes)

    plot_hashes(["base_maddpg", "ddpg"],
                "Comparison",
                "reward",
                save=save)


def main():
    plot_tragedy_commons()
    # plot_hunter_gatherer()


if __name__ == '__main__':
    main()
