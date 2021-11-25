import numpy as np
from multiagent.core import World, Agent, Landmark, Food
from multiagent.scenario import BaseScenario


def default_config():
    return {
        "ALGO_RED": "maddpg",
        "ALGO_BLUE": "maddpg",
        "GATHERERS_RED": 1,
        "HUNTERS_RED": 1,
        "GATHERERS_BLUE": 1,
        "HUNTERS_BLUE": 1,
        "FOOD_OBJECTS": 2,
        "LANDMARKS": 0,
        "P_ADV_COLLISION": 5.0,
        "CATCH_REWARD_HUNTERS": 20.0,
        "CATCH_REWARD_INDIVIDUAL": 5.0,
        "FOOD_REWARD_GATHERERS": 5.0,
        "FOOD_REWARD_INDIVIDUAL": 20.0,
        "DISTANCE_REWARD_MULTIPLIER": 0.1,
        "TEAM_REWARD_MULTIPLIER": 0.3,
        "P_INACTIVE": 1.0,
        "SPEED_HUNTERS": 1.0,
        "SPEED_GATHERERS": 1.0,
        "RESPAWN": False
    }


class Scenario(BaseScenario):
    def __init__(self, kwargs=None):


        self.GATHERERS_RED = 1
        self.HUNTERS_RED = 1
        self.GATHERERS_BLUE = 1
        self.HUNTERS_BLUE = 1
        self.FOOD_OBJECTS = 2
        self.LANDMARKS = 0

        # Algorithm modes
        self.ALGO_RED = "maddpg"
        self.ALGO_BLUE = "maddpg"

        # Penalty for gatherers if they are caught
        self.P_ADV_COLLISION = 5.0

        # reward for the hunters if they catches gatherers
        self.CATCH_REWARD_HUNTERS = 20.0
        self.CATCH_REWARD_INDIVIDUAL = 5.0

        # reward for the gatherers if they collect food
        self.FOOD_REWARD_GATHERERS = 5.0
        self.FOOD_REWARD_INDIVIDUAL = 20.0

        # reward for the gatherers if they collect food
        self.DISTANCE_REWARD_MULTIPLIER = 0.1

        # Multiplier for team rewards
        self.TEAM_REWARD_MULTIPLIER = 0.3

        # Punish inactiveness
        self.P_INACTIVE = 1.0

        # Speed
        self.SPEED_HUNTERS = 1.0
        self.SPEED_GATHERERS = 1.0

        # Respawn
        self.RESPAWN = False

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
        num_hunters_team_red = self.HUNTERS_RED
        num_gatherers_team_red = self.GATHERERS_RED
        num_hunters_team_blue = self.HUNTERS_BLUE
        num_gatherers_team_blue = self.GATHERERS_BLUE
        num_agents = num_hunters_team_red + num_gatherers_team_red + num_hunters_team_blue + num_gatherers_team_blue
        self.collectedBlue = 0 # count how many foods have been collected
        self.collectedRed = 0
        num_landmarks = self.LANDMARKS
        num_food = self.FOOD_OBJECTS

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_hunters_team_red:
                agent.hunter = True
                agent.gatherer = False
                agent.red = True
                agent.blue = False
                agent.algo_mode = self.ALGO_RED
                agent.agents_catched = 0
                agent.movable = True
            elif i < num_hunters_team_red + num_hunters_team_blue:
                agent.hunter = True
                agent.gatherer = False
                agent.red = False
                agent.blue = True
                agent.algo_mode = self.ALGO_BLUE
                agent.agents_catched = 0
                agent.movable = True
            elif i < num_hunters_team_red + num_hunters_team_blue + num_gatherers_team_blue:
                agent.hunter = False
                agent.gatherer = True
                agent.red = False
                agent.blue = True
                agent.algo_mode = self.ALGO_BLUE
                agent.food_collected = 0
                agent.movable = True
            elif i < num_hunters_team_blue + num_hunters_team_red + num_gatherers_team_blue + num_gatherers_team_red:
                agent.hunter = False
                agent.gatherer = True
                agent.red = True
                agent.blue = False
                agent.algo_mode = self.ALGO_RED
                agent.food_collected = 0
                agent.movable = True
            agent.max_speed = self.SPEED_HUNTERS if agent.hunter else self.SPEED_GATHERERS
            agent.size = 0.06 if agent.hunter else 0.04
            agent.accel = 3.0 if agent.hunter else 3.0

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.08
            landmark.boundary = False

        # add food
        world.food_objects = [Food() for i in range(num_food)]
        for i, food in enumerate(world.food_objects[:]):
            food.name = 'food %d' % i
            food.collide = True
            food.movable = False
            food.size = 0.02
            food.boundary = False

        # Color for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.1, 0.25, 0.5]) if agent.blue else np.array([0.8, 0.1, 0.1])
        # Color for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # Color for food
        for i, food in enumerate(world.food_objects[:]):
            food.color = np.array([0.35, 0.85, 0.35])

        # Set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
        for i, food in enumerate(world.food_objects):
            if not food.boundary:
                food.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                food.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        item_collected = 0
        if agent.gatherer:
            for i, food in enumerate(world.food_objects[:]):
                if agent.red:
                    #if self.is_collision(agent, food):
                    item_collected += self.collectedRed
                    self.collectedRed = 0
                if agent.blue:
                    #if self.is_collision(agent, food):
                    item_collected += self.collectedBlue
                    self.collectedBlue = 0
        else:
            if agent.red:
                for gatherer in self.gatherers_blue(world):
                    if self.is_collision(agent, gatherer):
                        item_collected += 1
            if agent.blue:
                for gatherer in self.gatherers_red(world):
                    if self.is_collision(agent, gatherer):
                        item_collected += 1
        return {"item_collected": item_collected}

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all red hunter agents
    def hunters_red(self, world):
        return [agent for agent in world.agents if agent.hunter and agent.red]

    # return all red gatherer agents
    def gatherers_red(self, world):
        return [agent for agent in world.agents if agent.gatherer and agent.red]

    # return all blue hunter agents
    def hunters_blue(self, world):
        return [agent for agent in world.agents if agent.hunter and agent.blue]

    # return all blue gatherer agents
    def gatherers_blue(self, world):
        return [agent for agent in world.agents if agent.gatherer and agent.blue]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.hunter_reward(agent, world) if agent.hunter else self.gatherer_reward(agent, world)
        return main_reward

    def gatherer_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        hunters_red = self.hunters_red(world)
        hunters_blue = self.hunters_blue(world)

        gatherers_blue = self.gatherers_blue(world)
        gatherers_red = self.gatherers_red(world)

        shape = True
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            if agent.red:
                for hunter in hunters_blue:
                    rew += self.DISTANCE_REWARD_MULTIPLIER * np.sqrt(
                        np.sum(np.square(agent.state.p_pos - hunter.state.p_pos)))
            if agent.blue:
                for hunter in hunters_red:
                    rew += self.DISTANCE_REWARD_MULTIPLIER * np.sqrt(
                        np.sum(np.square(agent.state.p_pos - hunter.state.p_pos)))
            # more reward if closer to food
            for food in world.food_objects[:]:
                rew += (-self.DISTANCE_REWARD_MULTIPLIER) * np.sqrt(
                    np.sum(np.square(agent.state.p_pos - food.state.p_pos)))

        # Negative reward for collision with adversary
        if agent.collide:
            if agent.red:
                for hunter in hunters_blue:
                    if self.is_collision(hunter, agent):
                        rew -= self.P_ADV_COLLISION
                        if self.RESPAWN:
                            agent.state.p_pos = np.random.uniform(-1, -0.9, world.dim_p)
            if agent.blue:
                for hunter in hunters_red:
                    if self.is_collision(hunter, agent):
                        rew -= self.P_ADV_COLLISION
                        if self.RESPAWN:
                            agent.state.p_pos = np.random.uniform(+0.9, +1, world.dim_p)

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

        for i, food in enumerate(world.food_objects[:]):
            if agent.red:
                for gatherer in gatherers_red:
                    if self.is_collision(gatherer, food):
                        rew += self.FOOD_REWARD_GATHERERS
                        food.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                        self.collectedRed += 1
                        if gatherer is agent:
                            rew += self.FOOD_REWARD_INDIVIDUAL
                            agent.food_collected += 1
            if agent.blue:
                for gatherer in gatherers_blue:
                    if self.is_collision(gatherer, food):
                        rew += self.FOOD_REWARD_GATHERERS
                        food.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                        self.collectedBlue += 1
                        if gatherer is agent:
                            rew += self.FOOD_REWARD_INDIVIDUAL
                            agent.food_collected += 1

        # Team reward
        if agent.red:
            all_agents_catched = 0
            for hunter in hunters_red:
                all_agents_catched += hunter.agents_catched
            rew += (all_agents_catched) * self.TEAM_REWARD_MULTIPLIER
        elif agent.blue:
            all_agents_catched = 0
            for hunter in hunters_blue:
                all_agents_catched += hunter.agents_catched
            rew += (all_agents_catched) * self.TEAM_REWARD_MULTIPLIER

        # Punish for inactiveness
        rew -= self.P_INACTIVE / (agent.food_collected + 1)

        return rew

    def hunter_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        gatherers_red = self.gatherers_red(world)
        gatherers_blue = self.gatherers_blue(world)
        hunters_red = self.hunters_red(world)
        hunters_blue = self.hunters_blue(world)

        shape = True
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            if agent.red:
                for gatherer in gatherers_blue:
                    # minimum distance so they want to minimize distance of closest hunter
                    rew -= self.DISTANCE_REWARD_MULTIPLIER * min(
                        [np.sqrt(np.sum(np.square(gatherer.state.p_pos - hunter.state.p_pos))) for hunter in
                         hunters_red])
                    # rew -= 0.1 * np.sqrt(np.sum(np.square(gatherer.state.p_pos - agent.state.p_pos)))
            if agent.blue:
                for gatherer in gatherers_red:
                    # minimum distance so they want to minimize distance of closest hunter
                    rew -= self.DISTANCE_REWARD_MULTIPLIER * min(
                        [np.sqrt(np.sum(np.square(gatherer.state.p_pos - hunter.state.p_pos))) for hunter in
                         hunters_blue])
                    # rew -= 0.1 * np.sqrt(np.sum(np.square(gatherer.state.p_pos - agent.state.p_pos)))

        # Positive reward for colliding with adversary
        if agent.collide:
            if agent.red:
                for gatherer in gatherers_blue:
                    for hunter in hunters_red:
                        if self.is_collision(hunter, gatherer):
                            rew += self.CATCH_REWARD_HUNTERS
                            if hunter is agent:
                                rew += self.CATCH_REWARD_INDIVIDUAL
                                agent.agents_catched += 1
            if agent.blue:
                for gatherer in gatherers_red:
                    for hunter in hunters_blue:
                        if self.is_collision(hunter, gatherer):
                            rew += self.CATCH_REWARD_HUNTERS
                            if hunter is agent:
                                rew += self.CATCH_REWARD_INDIVIDUAL
                                agent.agents_catched += 1

        # agents are penalized for exiting the screen
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        # Team reward
        if agent.red:
            all_food_collected = 0
            for gatherer in gatherers_red:
                all_food_collected += gatherer.food_collected
            rew += (all_food_collected) * self.TEAM_REWARD_MULTIPLIER
        elif agent.blue:
            all_food_collected = 0
            for gatherer in gatherers_blue:
                all_food_collected += gatherer.food_collected
            rew += (all_food_collected) * self.TEAM_REWARD_MULTIPLIER

        # Punish for inactiveness
        rew -= self.P_INACTIVE / (agent.agents_catched + 1)

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
            if not other.hunter:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
