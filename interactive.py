from time import sleep

import numpy as np
from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios


def main(scenario_name="tragedy_common"):
    def random_sample_step(nr_steps):
        one_hot = np.zeros((nr_steps,))
        # one_hot[np.random.randint(0, nr_steps)] = 1.0
        one_hot[0] = 1.0
        return one_hot

    scenario = scenarios.load(f"{scenario_name}.py").Scenario(None)

    world = scenario.make_world()
    n_agents = len(world.agents)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None,
                        shared_viewer=True)
    env.render()
    policy = InteractivePolicy(env, 0)
    obs_n = env.reset()
    while True:
        actions = [random_sample_step(5) for _ in range(n_agents)]  # TODO make 5 generic
        act_n = [policy.action(obs_n)] + actions
        obs_n, reward_n, done_n, _ = env.step(act_n)
        print(reward_n)

        sleep(0.5)
        env.render()


if __name__ == '__main__':
    main()
