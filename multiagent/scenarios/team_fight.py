import numpy as np
from multiagent.core import World, Agent, Landmark, Food
from multiagent.scenario import BaseScenario

def default_config():
    return {
        "ALGO_RED": "maddpg",
        "ALGO_BLUE": "maddpg",
        "AGENTS_RED": 2,
        "AGENTS_BLUE": 2,
        "FOOD_OBJECTS": 4,
        "LANDMARKS": 0,
        "P_ADV_COLLISION": 0.7,
        "CATCH_REWARD_GROUP": 20.0,
        "CATCH_REWARD_INDIVIDUAL": 5.0,
        "FOOD_REWARD_GROUP": 0.0,
        "FOOD_REWARD_INDIVIDUAL": 5.0,
        "DISTANCE_REWARD_MULTIPLIER": 0.1,
        "SECOND_AGENT_DISTANCE_MULTIPLIER": 0.3,
        "TEAM_REWARD_MULTIPLIER": 0.0,
        "P_INACTIVE": 0.0,
        "SPEED_RED": 1.0,
        "SPEED_BLUE": 1.0,
        "RESPAWN": True
    }

class Scenario(BaseScenario):
    def __init__(self, kwargs=None):


        self.AGENTS_RED = 2
        self.AGENTS_BLUE = 2
        self.FOOD_OBJECTS = 4
        self.LANDMARKS = 0

        # Algorithm modes
        self.ALGO_RED = "maddpg"
        self.ALGO_BLUE = "maddpg"

        # Penalty for agents if they are caught
        self.P_ADV_COLLISION = 0.7

        # reward for the agents if they catch others
        self.CATCH_REWARD_GROUP = 20.0
        self.CATCH_REWARD_INDIVIDUAL = 5.0

        # reward for the agents if they collect food
        self.FOOD_REWARD_GROUP = 0.0
        self.FOOD_REWARD_INDIVIDUAL = 5.0

        # reward for the distance
        self.DISTANCE_REWARD_MULTIPLIER = 0.1

        # reward for the distance, when there is another agents trying to catch an agent
        self.SECOND_AGENT_DISTANCE_MULTIPLIER = 0.3

        # Multiplier for team rewards
        self.TEAM_REWARD_MULTIPLIER = 0.0

        # Punish inactiveness
        self.P_INACTIVE = 0.0

        # Speed
        self.SPEED_RED = 1.0
        self.SPEED_BLUE = 1.0

        # Respawn
        self.RESPAWN = True

        if kwargs is not None:
            super().__init__(kwargs)
        else:
            super().__init__()

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        num_team_red = 2
        num_team_blue = 2
        num_agents = num_team_red + num_team_blue
        num_landmarks = self.LANDMARKS
        num_food = self.FOOD_OBJECTS

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.deleted = False
            agent.silent = True
            if i < num_team_red:
                agent.red = True
                agent.blue = False
                agent.algo_mode = self.ALGO_RED
                agent.state.p_pos = np.random.uniform(-0.9, -0.91, world.dim_p)
                agent.agents_catched = 0
                agent.food_collected = 0
                agent.times_caught = 0
                agent.max_speed = self.SPEED_RED
            elif i < num_team_red + num_team_blue:
                agent.red = False
                agent.blue = True
                agent.algo_mode = self.ALGO_BLUE
                agent.state.p_pos = np.random.uniform(+0.9, +0.91, world.dim_p)
                agent.agents_catched = 0
                agent.food_collected = 0
                agent.times_caught = 0
                agent.max_speed = self.SPEED_BLUE
            agent.size = 0.06
            agent.accel = 3.0
            agent.color = np.array([0.1, 0.25, 0.5]) if agent.blue else np.array([0.8, 0.1, 0.1])

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.08
            landmark.boundary = False
            landmark.color = np.array([0.25, 0.25, 0.25])
            # set random initial states
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

        # add food
        world.food_objects = [Food() for i in range(num_food)]
        for i, food in enumerate(world.food_objects[:]):
            food.name = 'food %d' % i
            food.collide = True
            food.movable = False
            food.size = 0.02
            food.boundary = False
            food.color = np.array([0.35, 0.85, 0.35])
            # set random initial states
            if not food.boundary:
                food.state.p_pos = np.random.uniform(-0.8, +0.8, world.dim_p)
                food.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        collisions = 0
        for a in self.agents_red(world):
            if self.is_collision(a, agent):
                collisions += 1
        for a in self.agents_blue(world):
            if self.is_collision(a, agent):
                collisions += 1
        return collisions


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all red agents
    def agents_red(self, world):
        return [agent for agent in world.agents if agent.red]

    # return all blue agents
    def agents_blue(self, world):
        return [agent for agent in world.agents if agent.blue]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        agents_red = self.agents_red(world)
        agents_blue = self.agents_blue(world)

        shape = True 
        if shape:  # reward can optionally be shaped (increased reward for decreased distance from food)
            for food in world.food_objects[:]:
                rew += (-self.DISTANCE_REWARD_MULTIPLIER) * np.sqrt(np.sum(np.square(agent.state.p_pos - food.state.p_pos)))
            if agent.red:
                for enemy in agents_blue:
                    for hunter in agents_red:
                        rew -= self.DISTANCE_REWARD_MULTIPLIER * np.sqrt(np.sum(np.square(enemy.state.p_pos - hunter.state.p_pos)))
            if agent.blue:
                for enemy in agents_red:
                    for hunter in agents_blue:
                        rew -= self.DISTANCE_REWARD_MULTIPLIER * np.sqrt(np.sum(np.square(enemy.state.p_pos - hunter.state.p_pos)))
            if agent.collide:
                if agent.red:
                    for ally in agents_red:
                        for enemy in agents_blue:
                            if self.is_collision(enemy, ally):
                                if ally is agent: continue
                                rew += (-self.SECOND_AGENT_DISTANCE_MULTIPLIER) * np.sqrt(np.sum(np.square(agent.state.p_pos - enemy.state.p_pos)))
                if agent.blue:
                    for ally in agents_blue:
                        for enemy in agents_red:
                            if self.is_collision(enemy, ally):
                                if ally is agent: continue
                                rew += (-self.SECOND_AGENT_DISTANCE_MULTIPLIER) * np.sqrt(np.sum(np.square(agent.state.p_pos - enemy.state.p_pos)))

        # Rewards for catching/getting caught
        if agent.collide:
            if agent.red:
                for enemy in agents_blue:
                    if self.is_collision(enemy, agent):
                        rew += self.CATCH_REWARD_INDIVIDUAL
                        for ally in agents_red:
                            if ally is agent: continue
                            if self.is_collision(enemy, ally):
                                rew += self.CATCH_REWARD_GROUP
                                agent.agents_catched += 1
                                enemy.times_caught += 1
                                if (self.RESPAWN):
                                    enemy.state.p_pos = np.random.uniform(+0.9, +1, world.dim_p)
            if agent.blue:
                for enemy in agents_red:
                    if self.is_collision(enemy, agent):
                        rew += self.CATCH_REWARD_INDIVIDUAL
                        for ally in agents_blue:
                            if ally is agent: continue
                            if self.is_collision(enemy, ally):
                                rew += self.CATCH_REWARD_GROUP
                                agent.agents_catched += 1
                                enemy.times_caught += 1
                                if (self.RESPAWN):
                                    enemy.state.p_pos = np.random.uniform(-1, -0.9, world.dim_p)

        rew -= self.P_ADV_COLLISION * agent.times_caught

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        #Food reward
        for i, food in enumerate(world.food_objects[:]):
            if agent.red:
                for red in agents_red:
                    if self.is_collision(red, food):
                        rew += self.FOOD_REWARD_GROUP
                        food.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                        if red is agent:
                            rew += self.FOOD_REWARD_INDIVIDUAL
                            agent.food_collected += 1
            if agent.blue:
                for blue in agents_blue:
                    if self.is_collision(blue, food):
                        rew += self.FOOD_REWARD_GROUP
                        food.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                        if blue is agent:
                            rew += self.FOOD_REWARD_INDIVIDUAL
                            agent.food_collected += 1

        # Team reward
        if agent.red:
            all_food_collected = 0
            all_agents_catched = 0
            for agent in agents_red:
                all_food_collected += agent.food_collected
                all_agents_catched += agent.agents_catched
            rew += (all_food_collected + all_agents_catched) * self.TEAM_REWARD_MULTIPLIER
        if agent.blue:
            all_food_collected = 0
            all_agents_catched = 0
            for agent in agents_blue:
                all_food_collected += agent.food_collected
                all_agents_catched += agent.agents_catched
            rew += (all_food_collected + all_agents_catched) * self.TEAM_REWARD_MULTIPLIER

        # Punish for inactiveness
        rew -= self.P_INACTIVE / (agent.food_collected + 1)

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        for entity in world.food_objects:
             if not entity.boundary:
                 entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # only agents, not adversaries communicate??
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
