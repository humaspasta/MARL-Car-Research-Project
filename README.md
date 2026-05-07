The Car project is a Multi-Agent Reinforcement Learning (MARL) project that seeks to maximize the total distance a car travels without crashing. The files here include all work that are relevant to creating the 
best network I have so far stored in best_network.txt.

The project specification is as follows:

The MARL car project seeks to maximize the distance traveled of a car in a confined, circular race track. The job of the agent is to compete for who can go the farthest. Crashes into other players and into the wall
are not penalized explicity but do reduce the speed of the car by 25%. The score is calculated and the distance traveled in a fixed number of frames. 

The action space for the purposes of the competition are assumed discrete. Therefore the agent must select from either turning left, turning right, speeding up, or do nothing given an input state. The do nothing
action results in the slow but forward movement of the car so do nothing does not directly imply nothing happens in the simulation. 

# 1 Data Collection

All data collection and formatting is done useing selenium which scrapes an environment that exists on an online website. Since the mode of learning is offline, we collect tabular data from the website (since the website is kind enough to us)
which are tuples, respresenting each state in a run, of the form (action , x , y vx, vy , sensor1, sensor2, sensor3,...sensor n) where n is an odd integer. Each sensor starts at 0 degrees on the circular agent and shoots a line radially outward from the circular agent and returns the distance to a target which is where that sensor line intersects with an object.

The lines are inputted as features to our model. 

The data collection step involves building the **agent.py** class. This class scrapes the website, inputs the networks values (which also handles formatting) and allows the simulation to run until 1000+ states are collected.
When the states appear in the box, the agent formats the data into (state, action, reward) tuples and returns the data as is. The pipelines summary is therefore

### network stored locally -> website -> data formatter -> formatted data. 

When the data is formatted we input the data into the model. Note that the reward is calculated within the agent.py class. 

# 1 Training 

The training involves three different pieces:

1) The model
2) the reward function
3) the training algorithm

# The Model

The model is a sequential neural network called DQN which is a misleading name because the same model is used for both DQN and Policy gradient methods. The model consists of the following layers

## Linear(n , 50) -> ReLu -> Linear(50 , 25) -> ReLu -> layer3(25 , 4) 

The output layer values correspond to the 4 values that are used to determine the decision.

# 2 Reward function

The reward function was created the way it is as a result of trial and error. One of the largest things that I struggled on in this project was finding a way to reward good decisions without indrecitly encouraging bad ones.
The current reward function works as follows.

Given a state $$s_t$$ and a corresponding 

