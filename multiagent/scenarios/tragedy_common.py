import json
import random
import time
from enum import Enum

import numpy as np
import math
from multiagent.core import World, Agent, Landmark, Food
from multiagent.scenario import BaseScenario
import uuid
import matplotlib.pyplot as plt

GREY = np.array([0.25, 0.25, 0.25])
GREEN_ALPHA = np.array([0.35, 0.85, 0.35, 0.2])
GREEN = np.array([0.35, 0.85, 0.35, 1.0])


def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r / 256.0, g / 256.0, b / 256.0))
    return ret


def get_cmap(n):
    out_ = [(0.39296875, 0.5703125, 0.93359375),
            (0.54296875, 0.3203125, 0.68359375),
            (0.55859375, 0.3359375, 0.19921875),
            (0.56640625, 0.34375, 0.70703125),
            (0.55078125, 0.328125, 0.69140625),
            (0.80078125, 0.578125, 0.94140625)]  # TODO this is currently hardcoded for rendering%

    return list(out_)


def generate_position(world, r=None):
    feeding_place = world.landmarks[0]
    if r is None:
        r = (feeding_place.size - 0.2) * np.sqrt(np.random.random())
    theta = np.random.random() * 2 * math.pi

    x = feeding_place.state.p_pos[0] + r * np.cos(theta)
    y = feeding_place.state.p_pos[1] + r * np.sin(theta)
    return np.asarray([x, y])


def landmark_collides(obj1, obj2):
    p_1, r_1 = obj1.state.p_pos, obj1.size
    p_2, r_2 = obj2.state.p_pos, obj2.size
    delta_pos = p_1 - p_2
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = r_1 + r_2
    return dist < dist_min


def generate_random_colors(world):
    cmap = get_cmap(len(world.agents))
    return np.asarray(cmap)[:len(world.agents)]


class FoodState(Enum):
    SPAWN = 0
    COLLECTED = 1


def _find_min_dist_food(world, agent):
    def dist_food(food):
        return np.linalg.norm(agent.state.p_pos - food.state.p_pos)

    min_food = world.food_objects[0]
    for food in world.food_objects:
        if dist_food(food) < dist_food(min_food) and food.c_state != FoodState.COLLECTED:
            min_food = food
    return min_food


def get_default_config():
    return {"AGENTS_NR": 3,
            "FOOD_OBJECTS": 3,
            "RESPAWN_TIMEOUT": 10,
            "P_FOOD_DIST": 10.0,
            "P_TIME": 5.0,
            "P_DESERT": 500.0,
            "COLLECTION_REWARD_DIST": 0.1
            }


class Scenario(BaseScenario):
    def __init__(self, kwargs):
        self.AGENTS_NR = 3
        self.FOOD_OBJECTS = 3
        self.RESPAWN_TIMEOUT = 10

        # Penalties in order to smooth env gradient
        self.P_FOOD_DIST = 20.0
        self.P_TIME = 10.0

        # Penalty if the enviroment gets 'deactivated' (all foods where collected)
        self.P_DESERT = 500.0

        # env specific
        self.FOREST_RADIUS = 0.4

        # pr collection_reward for pr=0 all reward will be granted at home
        #                      for pr=1 all reward will be granted during the food collection
        self.COLLECTION_REWARD_DIST = 0.5
        self.OVERALL_COLLECTION_REWARD = 40.0
        self.INDV_COLLECTION_BONUS = self.COLLECTION_REWARD_DIST * self.OVERALL_COLLECTION_REWARD
        self.INDV_HOME_BONUS = (1 - self.COLLECTION_REWARD_DIST) * self.OVERALL_COLLECTION_REWARD

        # overall reward for collecting food
        self.COLLECTION_BONUS = 0.0

        self.ALGO_MODES = ["maddpg" for _ in range(self.AGENTS_NR)]

        super().__init__(kwargs)

        # initalize data for benchmarks
        self.agents_collection_data = [0 for _ in range(self.AGENTS_NR)]
        self.agents_collection_data_home = [0 for _ in range(self.AGENTS_NR)]

    def _get_non_blocked_food(self, world):
        collected_food_objets = [food_obj for food_obj in world.food_objects if
                                 (food_obj.c_state == FoodState.COLLECTED
                                  and abs(food_obj.recreation_time - world.time) > self.RESPAWN_TIMEOUT)]
        if len(collected_food_objets) == 0:
            return None
        tmp_food = collected_food_objets[random.randint(0, len(collected_food_objets) - 1)]
        i = 0
        for agent in world.agents:
            if landmark_collides(agent, tmp_food):
                i += 1
                if i >= len(collected_food_objets):
                    return None
                tmp_food = collected_food_objets[i]
        return tmp_food

    def step_callback(self, world):
        for i, agent in enumerate(world.agents):
            for food_obj in world.food_objects:
                if landmark_collides(agent, food_obj) and food_obj.c_state == FoodState.SPAWN:
                    agent.food_collect += 1
                    agent.live_pts += self.INDV_COLLECTION_BONUS
                    self.agents_collection_data[i] += 1
                    for other_agent in world.agents:
                        other_agent.live_pts += self.COLLECTION_BONUS

                    food_obj.c_state = FoodState.COLLECTED
                    food_obj.recreation_time = int(world.time)
                    self._update_color(food_obj)
            home = world.landmarks[i + 1]

            if landmark_collides(agent, home) and agent.food_collect != 0:
                agent.live_pts += self.INDV_HOME_BONUS
                self.agents_collection_data_home[i] += 1
                agent.food_collect = 0

        active_food_objects = [food_obj for food_obj in world.food_objects if food_obj.c_state == FoodState.SPAWN]

        if len(active_food_objects) != 0:
            tmp_food = self._get_non_blocked_food(world)
            if tmp_food:
                tmp_food.c_state = FoodState.SPAWN
                self._update_color(tmp_food)

    def _update_color(self, food):
        if food.c_state == FoodState.SPAWN:
            food.color = GREEN
        else:
            food.color = GREY

    def make_world(self):
        world = World()
        world.step_callback = self.step_callback

        # add landmarks
        world.landmarks = [Landmark() for _ in range(1 + self.AGENTS_NR)]
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.name = 'Forest'
                landmark.collide = False
                landmark.movable = False
                landmark.size = self.FOREST_RADIUS
            else:
                landmark.name = f'Barn {i}'
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.075

        # add agents
        world.agents = [Agent() for _ in range(self.AGENTS_NR)]
        for i, agent in enumerate(world.agents):
            agent.name = 'Agent %d' % i
            agent.collide = True
            agent.accel = 3.0
            agent.algo_mode = self.ALGO_MODES[i]
            agent.silent = True

        world.food_objects = []
        for i in range(self.FOOD_OBJECTS):
            food = Food()
            food.color = GREY
            food.name = f"Food {i}"
            food.collide = False
            food.movable = False
            food.size = 0.02
            food.boundary = False
            food.c_state = FoodState.SPAWN
            food.agent_collected_idx = None
            world.food_objects.append(food)

        self.reset_world(world)
        return world

    def reset_world(self, world):
        random_colors = generate_random_colors(world)
        world.time = 0

        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = GREEN_ALPHA
                landmark.state.p_pos = np.zeros(world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.color = random_colors[i - 1]
                landmark.state.p_pos = generate_position(world, self.FOREST_RADIUS + 0.1)
                landmark.state.p_vel = np.zeros(world.dim_p)

        for food in world.food_objects:
            food.c_state = FoodState.SPAWN
            food.state.p_pos = generate_position(world)
            food.state.p_vel = np.zeros(world.dim_p)
            self._update_color(food)

        # set random initial states
        for agent in world.agents:
            i = world.agents.index(agent)
            agent.color = random_colors[i]
            agent.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.live_pts = 0
            agent.food_collect = 0

            # initalize data for benchmarks
        self.agents_collection_data = [0 for _ in range(self.AGENTS_NR)]
        self.agents_collection_data_home = [0 for _ in range(self.AGENTS_NR)]

    def reward(self, agent, world):
        agent_idx = world.agents.index(agent)
        assert agent_idx != -1
        rew = agent.live_pts

        if agent.food_collect == 0:
            food = _find_min_dist_food(world, agent)
            rew += (-self.P_FOOD_DIST) * (np.linalg.norm(agent.state.p_pos - food.state.p_pos)) / \
                   np.linalg.norm([2, 2])
        else:
            home = world.landmarks[agent_idx + 1]
            rew += (-self.P_FOOD_DIST) * (np.linalg.norm(agent.state.p_pos - home.state.p_pos)) / \
                   np.linalg.norm([2, 2])

        active_food_objects = [food_obj for food_obj in world.food_objects if food_obj.c_state == FoodState.SPAWN]
        if len(active_food_objects) == 0:
            rew -= self.P_DESERT

        rew += (-self.P_TIME) * world.time / 100.0

        def bound(x):
            if x < 1.0:
                return 0
            return 10.0 * min(np.exp(2 * x - 2), 10)

        penality = 0
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            penality -= bound(x)

        rew += penality
        return rew

    def observation(self, agent, world):
        agent_idx = world.agents.index(agent)
        agent_pos = []
        for agent_ in world.agents:
            agent_pos.append(agent_.state.p_pos)
        home_pos = (world.landmarks[agent_idx + 1].state.p_pos - agent.state.p_pos)
        # display the food position if its available and not collected
        food_pos = []
        food_state = []
        for food in world.food_objects:
            food_pos.append(food.state.p_pos - agent.state.p_pos)
            food_state.append((int(food.c_state is FoodState.SPAWN)))

        positions = np.concatenate(agent_pos + food_pos + [home_pos])
        return np.asarray(list(positions) + food_state + [agent.food_collect]).flatten()

    def benchmark_data(self, agent, world):
        agent_idx = world.agents.index(agent)

        active_food_objects = [food_obj for food_obj in world.food_objects if food_obj.c_state == FoodState.SPAWN]
        tmp = {"agent_collected": self.agents_collection_data[agent_idx],
               "agent_collected_home": self.agents_collection_data_home[agent_idx],
               "still_active_foods": len(active_food_objects) == 0}
        self.agents_collection_data[agent_idx] = 0
        self.agents_collection_data_home[agent_idx] = 0
        return tmp
