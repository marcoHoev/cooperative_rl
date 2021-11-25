import torch
import numpy as np
import agent
import ddpg
import random
from gym.spaces import Box

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class MADDPG:
    def __init__(self, actors, hash, arglist):
        self.workers = []
        self.actors = actors
        self.hash = hash
        # exploration
        self.explore = arglist.epsilon
        pass

    def _algo_mode_from_agents(self, env):
        algo_mode = []

        for agent in env.get_env().agents:
            if agent.algo_mode == "ddpg":  # adversary
                algo_mode.append(ddpg.DDPG)  # MADDPG
            elif agent.algo_mode == "maddpg":
                algo_mode.append(ddpg.MADDPG)
            elif agent.algo_mode == "demaddpg":
                algo_mode.append(ddpg.DEMADDPG)
            else:
                raise Exception(f"Unknown Alorithm mode: {agent.algo_mode}")
        return algo_mode

    def create_agents(self, env, arglist, root="saved", hash=None):
        # workers = []
        if hash is None:
            hash = arglist.scenario
        algo_mode = self._algo_mode_from_agents(env)

        obs_shapes = [env.get_env().observation_space[i].shape for i in range(env.get_env().n)]
        actions_shape_n = [env.get_env().action_space[i].n for i in range(env.get_env().n)]
        actions_n = 0
        obs_shape_n = 0
        for actions in actions_shape_n:
            actions_n += actions
        for obs_shape in obs_shapes:
            obs_shape_n += obs_shape[0]

        for i, action_space, observation_space, algo in zip(range(len(env.get_env().action_space)),
                                                            env.get_env().action_space,
                                                            env.get_env().observation_space,
                                                            algo_mode):
            if isinstance(action_space, Box):
                discrete_action = False
            else:
                discrete_action = True

            if algo == ddpg.DEMADDPG:
                print('DE-MADDPG load.')
                local_critic = agent.Critic(observation_space.shape[0], action_space.n).to(device)
                actor = agent.Actor(observation_space.shape[0], action_space.n).to(device)
                target_local_critic = agent.Critic(observation_space.shape[0], action_space.n, arglist.tau).to(device)
                target_actor = agent.Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)
                actor.eval()
                local_critic.eval()
                target_actor.eval()
                target_local_critic.eval()
                if i == 0:  # only one global critic
                    print('Initialized global critic.')
                    global_critic = agent.Critic(obs_shape_n, actions_n).to(device)
                    target_global_critic = agent.Critic(obs_shape_n, actions_n, arglist.tau).to(device)
                    global_critic.eval()
                    target_global_critic.eval()

                ddpg_algo = ddpg.DEMADDPG(i, actor, local_critic, global_critic, target_actor, target_local_critic,
                                          target_global_critic, arglist.gamma,
                                          arglist.batch_size, arglist.eval, discrete_action, alg_mode='DEMADDPG')

                ddpg_algo.load(f'./{root}/{hash}/actor' + str(i) + '_' + str(arglist.load_episode_saved),
                               f'./{root}/{hash}/loc_critic' + str(i) + '_' + str(arglist.load_episode_saved),
                               f'./{root}/{hash}/glob_critic' + '_' + str(arglist.load_episode_saved))
            else:
                if algo == ddpg.MADDPG:
                    print('MADDPG load.')
                    critic = agent.Critic(obs_shape_n, actions_n).to(device)
                    actor = agent.Actor(observation_space.shape[0], action_space.n).to(device)
                    target_critic = agent.Critic(obs_shape_n, actions_n, arglist.tau).to(device)
                    target_actor = agent.Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)
                else:  # DDPG
                    print('DDPG load.')
                    critic = agent.Critic(observation_space.shape[0], action_space.n).to(device)
                    actor = agent.Actor(observation_space.shape[0], action_space.n).to(device)
                    target_critic = agent.Critic(observation_space.shape[0], action_space.n, arglist.tau).to(device)
                    target_actor = agent.Actor(observation_space.shape[0], action_space.n, arglist.tau).to(device)

                actor.eval()
                critic.eval()
                target_actor.eval()
                target_critic.eval()

                ddpg_algo = ddpg.DDPG(i, actor, critic, target_actor, target_critic, arglist.gamma, arglist.batch_size,
                                      arglist.eval, discrete_action, alg_mode=algo)
                ddpg_algo.load(f'./{root}/{hash}/actor' + str(i) + '_' + str(arglist.load_episode_saved),
                               f'./{root}/{hash}/critic' + str(i) + '_' + str(arglist.load_episode_saved))
            self.workers.append(ddpg_algo)

    def reset(self):
        self.ep_ave_max_q_value = [0 for _ in self.workers]
        self.network_update = False
        for worker in self.workers:
            worker.ou.reset()

    def take_actions(self, obs, at_random=False, eps=0.0):
        actor_actions = []
        for actor_obs in obs:
            actions = []
            for i, worker in enumerate(self.workers):
                action = worker.act(actor_obs[i], at_random, eps=eps)
                actions.append(action)
            actor_actions.append(actions)
        return actor_actions

    def update(self, step, actors, actions, reward, obs, obs2, done):
        for actor in range(self.actors):
            for i, worker in enumerate(self.workers):
                worker.add(actions[actor][i], [reward[actor][i]], obs[actor][i], obs2[actor][i], [done[actor][i]])
        if step % 100 < actors:
            self.network_update = True
            update_target = False
            for i, worker in enumerate(self.workers):
                ep_avg_max_q, update_target = worker.train(self.workers)
                self.ep_ave_max_q_value[i] += ep_avg_max_q
            if update_target:
                for i, worker in enumerate(self.workers):
                    worker.update_targets()

    def save(self, episode, root="saved", logger=None):
        for worker in self.workers:
            if worker.alg_mode == 'DEMADDPG':
                worker.save(f'./{root}/{self.hash}/actor' + str(worker.pos) + '_' + str(episode),
                            f'./{root}/{self.hash}/loc_critic' + str(worker.pos) + '_' + str(episode),
                            f'./{root}/{self.hash}/glob_critic' + '_' + str(episode), logger=logger)
            else:
                worker.save(f'./{root}/{self.hash}/actor' + str(worker.pos) + '_' + str(episode),
                            f'./{root}/{self.hash}/critic' + str(worker.pos) + '_' + str(episode), logger=logger)
