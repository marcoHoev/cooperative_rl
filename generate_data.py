import json
from copy import deepcopy

from multiagent.scenarios.tragedy_common import get_default_config

DEFAULT_ARGS = {"scenario": "tragedy_common",
                "eval": True,
                "epsilon": 0.05,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.01,
                "load_episode_saved": 100000,
                "saved_episode": 100,
                "batch_size": 1024,
                "max_episode": 80001,
                "gamma": 0.95,
                "tau": 0.01,
                "plot": True,
                "plot_mode": "save"
                }

NR_FOOD = [2, 3, 4]
P_DESERTS = [0, 250, 500]
COLLECTION_REWARD_DIST = [0.1, 0.5, 0.9]
ALGS = ["ddpg", "maddpg"]


def generate_configs(alg):
    configs = []
    env_cfg = get_default_config()
    env_cfg["ALGO_MODES"] = [alg for _ in range(env_cfg["AGENTS_NR"])]
    for nr_food in NR_FOOD:
        hash_ = f"nrfood_{nr_food}_{alg}"
        env_cfg_ = deepcopy(env_cfg)
        env_cfg_["FOOD_OBJECTS"] = nr_food
        cfg = {
            "hash": hash_,
            "args": DEFAULT_ARGS,
            "env_config": env_cfg_
        }
        configs.append(cfg)
    for p_desert in P_DESERTS:
        hash_ = f"desert_{int(p_desert)}_{alg}"
        env_cfg_ = deepcopy(env_cfg)
        env_cfg_["P_DESERT"] = p_desert
        cfg = {
            "hash": hash_,
            "args": DEFAULT_ARGS,
            "env_config": env_cfg_
        }
        configs.append(cfg)
    for collection_rew in COLLECTION_REWARD_DIST:
        hash_ = f"prew_{int(100 * collection_rew)}_{alg}"
        env_cfg_ = deepcopy(env_cfg)
        env_cfg_["COLLECTION_REWARD_DIST"] = collection_rew
        cfg = {
            "hash": hash_,
            "args": DEFAULT_ARGS,
            "env_config": env_cfg_
        }
        configs.append(cfg)
    return configs


def main():
    configs = []
    for alg in ALGS:
        cfgs = generate_configs(alg)
        configs = configs + cfgs
    print(len(configs))
    with open("test.json", "w") as f:
        json.dump(configs, f)


def test():
    with open("exp_config.json", "r") as f:
        config = json.load(f)
    for idx in range(35, 65):
        print(idx, config[idx]["hash"])


if __name__ == '__main__':
    test()
