# Cooperative and competitive RL agents

This Project aims to compare and analyze cooperative and competitive strategies in RL.

## Setup

In order to run this project you need Pipenv: `pip install pipenv`.

Secondly run: `pipenv install`.

Make sure to either have python3.7 or pyenv ([link](https://www.liquidweb.com/kb/how-to-install-pyenv-on-ubuntu-18-04/)) installed, so that the setup with pipenv works smoothly. Also, modify your PATH variable accordingly, so that the packages belonging to pipenv are found correctly.

To train one of the scenarios with base configurations, run:

`pipenv run python train.py --scenario [name_of_scenario]`

Then in order to test the scenario, run:
`pipenv run python train.py --scenario [name_of_scenario] --eval --load-episode-saved [episode_number]`

Keep in mind to only test episode numbers that were saved, otherwise the agents will usually just move in straight lines, as they haven't been trained.

## Scenarios

There are different multi-agent scenarios to test cooperation and competition of RL agents. Several scenarios have been made and tested by 
openai: https://github.com/openai/multiagent-particle-envs. Check them out!
We have adopted their simple_tag environment and based on it, we created 3 more complex environments to further test cooperation and competition 
through reinforcement learning:

* tragedy_common
* hunter_gatherer_teams
* team_fight
* simple_tag

### Tragedy Commons Environment
This enviroment models the tragedy of commons scenario.
One can find the following parameters:

| Name                      | Description                                                                                                                                                                          | Default Values |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| AGENTS_NR                 | The number of agents in the enviroment                                                                                                                                               | 3              |
| FOOD_OBJECT               | The number of food elements in the enviroment                                                                                                                                        | 3              |
| RESPAWN_TIMEOUT           | How long a food element is inactive after it is eaten                                                                                                                                | 10             |
| P_FOOD_DIST               | The penalty thats getting added towards the distance to the nearest food element/home in order to smooth the gradient                                                                | 20             |
| P_TIME                    | With increasing time in the enviroment this reward grows in order to incentivse movement of the agents                                                                               | 10             |
| P_DESERT                  | This penalty will get subtracted from the overall reward every round, in case all food orbs had been eaten.                                                                          | 500            |
| COLLECTION_REWARD_DIST    | The distribution of the collection reward. A value $\in [0:1]$, it indicates how much reward is given for eating the food and how much reward is granted for bringing the food home. | 0.5            |
| OVERALL_COLLECTION_REWARD | The collection reward. How much is granted for collecting a food element and bringing it home.                                                                                       | 40             |
| ALGO_MODES                 | A list, that defines the algorithm of each agent | ["maddpg", "maddpg", "maddpg"]

### Hunter Gatherer Teams Environment

This environment is built of two teams, that consist of hunters and gatherers. The gatherers of a team try to collect food items, while the hunters try
to catch the gatherers of the opposite team. 
The parameters we introduced are:

| Name                       | Description                                                                                                                                                                                             | Default Values |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| ALGO_RED                   | Change the algorithm of the red team to either "ddpg", "maddpg" or "demaddpg"                                                                                                                           | maddpg         |
| ALGO_BLUE                  | Change the algorithm of the blue team to either "ddpg", "maddpg" or "demaddpg"                                                                                                                          | maddpg         |
| GATHERERS_RED              | Number of red team gatherers                                                                                                                                                                            | 1              |
| HUNTERS_RED                | Number of red team hunters                                                                                                                                                                              | 1              |
| GATHERERS_BLUE             | Number of blue team gatherers                                                                                                                                                                           | 1              |
| HUNTERS_BLUE               | Number of blue team hunters                                                                                                                                                                             | 1              |
| FOOD_OBJECTS               | Number of food objects                                                                                                                                                                                  | 2              |
| LANDMARKS                  | Number of randomly placed landmarks that the agents can collide with.                                                                                                                                   | 0              |
| P_ADV_COLLISION            | Penalty for individual gatherers when caught by hunters.                                                                                                                                                | 5.0            |
| CATCH_REWARD_HUNTERS       | Reward all hunters of a team get when one of them catches a gatherer of the opposite team                                                                                                               | 20.0           |
| CATCH_REWARD_INDIVIDUAL    | Individual reward a hunters gets for catching a gatherer of the opposite team.                                                                                                                          | 5.0            |
| FOOD_REWARD_GATHERERS      | Reward all gatherers of a team get when one of them collects a food item.                                                                                                                               | 5.0            |
| FOOD_REWARD_INDIVIDUAL     | Individual reward a gatherer gets for collecting a food item.                                                                                                                                           | 20.0           |
| DISTANCE_REWARD_MULTIPLIER | A parameter to change the rewards for the agents' distance to either food or other agents.                                                                                                              | 0.1            |
| TEAM_REWARD_MULTIPLIER     | A parameter to change the importance of team rewards, e.g. the rewards hunters get when their teams'  gatherers collect food items or the gatherers get when hunters catch gatherers of the other team. | 0.3            |
| P_INACTIVE                 | Individual penalty for being inactive                                                                                                                                                                   | 1.0            |
| SPEED_HUNTERS              | Speed of hunters                                                                                                                                                                                        | 1.0            |
| SPEED_GATHERERS            | Speed of gatherers                                                                                                                                                                                      | 1.0            |
| RESPAWN                    | Flag for setting respawning                                                                                                                                                                             | False          |
### Team Fight Environment

This environment is heavily based on hunter gatherer, as now the role of a hunter and gatherer is combined in one agents. Two agents of the same team can hunt an agent of the opposite team and all of the agents can collect food items
The parameters we introduced are:

| Name                       | Description                                                                                                                                                                                             | Default Values |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| ALGO_RED                   | Change the algorithm of the red team to either "ddpg", "maddpg" or "demaddpg"                                                                                                                           | maddpg         |
| ALGO_BLUE                  | Change the algorithm of the blue team to either "ddpg", "maddpg" or "demaddpg"                                                                                                                          | maddpg         |
| AGENTS_RED              | Number of red team agents                                                                                                                                                                            | 2              |
| AGENTS_BLUE             | Number of blue team agents                                                                                                                                                                           | 2              |
| HUNTERS_BLUE               | Number of blue team hunters                                                                                                                                                                             | 1              |
| FOOD_OBJECTS               | Number of food objects                                                                                                                                                                                  | 4              |
| LANDMARKS                  | Number of randomly placed landmarks that the agents can collide with.                                                                                                                                   | 0              |
| P_ADV_COLLISION            | Penalty for individual agents when caught                                                                                                                                                | 0.7            |
| CATCH_REWARD_GROUP       | Reward for a group when catching an agents of the opposite team                                                                                                               | 20.0           |
| CATCH_REWARD_INDIVIDUAL    | Individual reward an gets for catching an agent of the opposite team.                                                                                                                          | 5.0            |
| FOOD_REWARD_GROUP      | Reward all agents of a team get when one of them collects a food item.                                                                                                                               | 5.0            |
| FOOD_REWARD_INDIVIDUAL     | Individual reward an agent gets for collecting a food item.                                                                                                                                           | 20.0           |
| DISTANCE_REWARD_MULTIPLIER | A parameter to change the rewards for the agents' distance to either food or other agents.                                                                                                              | 0.1            |
| SECOND_AGENT_DISTANCE_MULTIPLIER | A parameter to change the rewards for the agents' distance when a teammate is trying to catch an agent.                                                                                                              | 0.3            |
| TEAM_REWARD_MULTIPLIER     | A parameter to change the importance of team rewards, e.g. the rewards hunters get when their teams'  gatherers collect food items or the gatherers get when hunters catch gatherers of the other team. | 0.3            |
| P_INACTIVE                 | Individual penalty for being inactive                                                                                                                                                                   | 1.0            |
| SPEED_RED              | Speed of red agents                                                                                                                                                                                        | 1.0            |
| SPEED_BLUE            | Speed of blue agents                                                                                                                                                                                      | 1.0            |
| RESPAWN                    | Flag for setting respawning                                                                                                                                                                             | True          |

## Run multiple experiments simultaneously
In order to run multiple experiments at a time run the `sh run_tests.sh`. This will launch a process for every experiment with the corresponding index of the `exp_config.json`. In this file you can specify a hash of the experiment and all parameters needed. When the experiments are finished, you will find the corresponding weights of actors and critics under ./experiments/[name_of_scenario]/[name_of_hash]. For the automated creation of the exp_config.json file for the scenario Tragedy Commons you can have a look at generate_data.py.

## Benchmarking 

In the implementations of each scenario you can find the function benchmark_data(). This is used to get meaningful benchmark data that goes beyond the reward of the agents. In the function plot.py we have set up everything needed for benchmarking:

- In the main() function uncomment the line depending on which scenario you want to evaluate
- In the corresponding plot_[name_of_scenario] function you can set two arguments:
  - generate_data: If True, for all the specified hashes in the function the benchmark data is generated (in ./experiments/[name_of_scenario]/[name_of_hash]/benchmark/) by loading the respective weights of the hash and running 1000 episodes with the learned policies. Note that you need to have the agents trained already, i.e. for each hash that you want to benchmark, you need to have carried out the experiments (see above) before! If you have generated the benchmark data already and just want to show the plot again, you can set generate_data = False
  - save: If true, the corresponding plots are saved in the folder ./plots. If False, you can just view the plots.

## Visualize environments

To interactively check out an environment, run:

`pipenv run bin/interactive.py --scenario [name_of_scenario].py`




## Additional Info

This repo is based on the following two other repositories:

* https://github.com/Gouet/MADDPG-pytorch
* https://github.com/openai/multiagent-particle-envs

Props and big thanks to them, they made our lives a little easier :) <3