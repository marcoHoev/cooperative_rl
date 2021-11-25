from time import sleep

import numpy as np
from scipy import randn

from multiagent import scenarios
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy, Policy


def is_collision(self, agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


def is_blocking(foods, food_states, direction_vecs, max_r=0.07):
    blocking = False
    for i, food_pos in enumerate(foods):
        if food_states[i] == 0:
            continue
        cut_counter = 0
        for direction_vec in direction_vecs:
            dist = direction_vec.T @ np.asarray(food_pos)
            if 0 < dist < max_r:
                cut_counter += 1
        if cut_counter > 1:
            blocking = True
    return blocking


def get_move_idx(food_pos, ):
    up = np.asarray([1, 0]).T @ np.asarray(food_pos)
    down = np.asarray([-1, 0]).T @ np.asarray(food_pos)
    left = np.asarray([0, 1]).T @ np.asarray(food_pos)
    right = np.asarray([0, -1]).T @ np.asarray(food_pos)
    return 1 + np.argmax([up, down, left, right])


def get_move_idx_blocking(food_pos, food_states, foods):
    up = np.asarray([1, 0]).T @ np.asarray(food_pos)
    if is_blocking(foods, food_states, [np.asarray([1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]):
        up = -1

    down = np.asarray([-1, 0]).T @ np.asarray(food_pos)
    if is_blocking(foods, food_states, [np.asarray([-1, 0]), np.asarray([0, 1]), np.asarray([0, -1])]):
        down = -1

    left = np.asarray([0, 1]).T @ np.asarray(food_pos)
    if is_blocking(foods, food_states, [np.asarray([0, 1]), np.asarray([1, 0]), np.asarray([-1, 0])]):
        left = -1

    right = np.asarray([0, -1]).T @ np.asarray(food_pos)
    if is_blocking(foods, food_states, [np.asarray([0, -1]), np.asarray([1, 0]), np.asarray([-1, 0])]):
        right = -1
    if np.all([up == -1, down == -1, left == -1, right == -1]):
        return 0
    return 1 + np.argmax([up, down, left, right])


class OptimalPolicyTragedyCommon(Policy):
    def __init__(self, n_agents, n_food, n_steps, agent_idx):
        super().__init__()
        self.n_food = n_food
        self.n_agents = n_agents
        self.food_obj = 0
        self.n_steps = n_steps
        self.agent_idx = agent_idx

    def transform_observsation(self, observation):
        offset = 2 * self.n_agents
        foods = np.reshape(observation[offset: offset + 2 * self.n_food], (self.n_food, 2))
        offset += 2 * self.n_food
        home_pos = observation[offset: offset + 2]
        offset += 2
        food_states = observation[offset: offset + self.n_food]
        food_count = observation[-1]
        return foods, food_states, home_pos, food_count

    def get_closest_food(self, foods, food_state):
        min_food = foods[0]
        for i, food in enumerate(foods):
            if np.linalg.norm(food) <= np.linalg.norm(min_food) and food_state[i] == 1.0:
                min_food = food
        return min_food

    def action(self, observation):

        foods, food_states, home_pos, food_count = self.transform_observsation(observation)
        out_dir = np.zeros((self.n_steps,))
        if food_count == 0:
            if self.agent_idx < 2 and randn() < 0.5:
                out_dir[0] = 1.0
                return out_dir
            if np.sum(food_states) < 3:
                out_dir[0] = 1.0
            else:
                food_pos = self.get_closest_food(foods, food_states)
                out_dir[get_move_idx(food_pos)] = 1.0
        else:
            out_dir[get_move_idx_blocking(home_pos, food_states, foods)] = 1.0
        return out_dir


def eval_scenario(scenario_name="tragedy_common", render=False):
    scenario = scenarios.load(f"{scenario_name}.py").Scenario(None)

    world = scenario.make_world()
    n_agents = len(world.agents)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                        shared_viewer=True)
    while render:
        if render:
            env.render()
        policies = [OptimalPolicyTragedyCommon(len(world.agents), len(world.food_objects), 5, i) for i in
                    range(n_agents)]
        obs_n = env.reset()
        step = 0
        reward = None
        while step < 100:
            act_n = [policies[i].action(obs) for i, obs in enumerate(obs_n)]
            obs_n, reward_n, done_n, _ = env.step(act_n)
            # print(reward_n)
            reward = reward_n

            if render:
                sleep(0.07)
                env.render()
            step += 1
        print(reward)


def main():
    reward = eval_scenario(render=True)
    print(reward)


if __name__ == '__main__':
    main()
