The Car project is a Multi-Agent Reinforcement Learning (MARL) project that seeks to maximize the total distance a car travels without crashing. The files here include all work that are relevant to creating the 
best network I have so far stored in best_network.txt.

The project specification is as follows:

The MARL car project seeks to maximize the distance traveled of a car in a confined, circular race track. The job of the agent is to compete for who can go the farthest. Crashes into other players and into the wall
are not penalized explicity but do reduce the speed of the car by 25%. The score is calculated and the distance traveled in a fixed number of frames. 

The action space for the purposes of the competition are assumed discrete. Therefore the agent must select from either turning left, turning right, speeding up, or do nothing given an input state. The do nothing
action results in the slow but forward movement of the car so do nothing does not directly imply nothing happens in the simulation. 

# 1 Data Collection

All data collection and formatting is done useing selenium which scrapes an environment that exists on an online website. Since the mode of learning is offline, we collect tabular data from the website in the form of tuples. For each time step $$t$$, the website returns a tuple of the form $$(action , x , y vx, vy , sensor1, sensor2, sensor3,...sensor n)$$ where n is an odd integer from some $$T > 0$$ time total steps. Each sensor is a line that shoots radially outward from the agent and returns the distance from the agent to where the line intersects with an object (another car or the track walls).

The data collection step involves building the **agent.py** class. This class scrapes the website, inputs the networks values (with correct formatting) and allows the simulation to run until 1000+ steps are collected. 
When the states appear in the box, the agent formats the data into (state, action, reward) tuples and returns the data as is. The pipelines summary is therefore

### agent.py -> website -> data formatter (in agent.py) -> formatted data. 

When the data is formatted we input the data into the model. Note that the reward is calculated within the agent.py class. 

# 1 Training 

The training involves three different pieces:

1) The model
2) the reward function
3) the training algorithm

# The Model

The model is a sequential neural network, the same model is used for both DQN and Policy gradient methods. The model consists of the following layers

## Linear(n , 50) -> ReLu -> Linear(50 , 25) -> ReLu -> Linear(25 , 4) 

The output layer values correspond to the 4 possible actions in the action space. 

# 2 Reward function

The reward function was created the way it is as a result of trial and error. One of the largest things that I struggled on in this project was finding a way to reward good decisions without indrecitly encouraging bad ones. Some examples of degenerative policies that my model found was either perpetually spinning policy (maximizes sensor values) or a speed up only policy (maximizes distance traveled. Ignores collisions). 


The current reward function works as follows.

Given a state $$s_t$$, corresponding state $s_{t+1}$, initial state $$s_0$$, and reward function $$R(t):S^3 \rightarrow \mathbb{R}$$

$$
  \huge R(s_t , s_{t+1}, s_0) = \alpha* d(x_t , y_t , x_t{t+1} , y_{t+1}) + \beta S(vx_{t+1} , vy_{t+1} , k) + \gamma A(vx_{t} , vy_{t}, vx_{t+1}, vy_{t+1}) + \nu P(sensor1, sensor2,...,sensork)
$$

where we define functions d, S, A, and P as:

- $$d(x_t , y_t ,  x_{t+1} , y_{t+1})$$: The standard eucldian distanec function.
    - $$\huge d(x_t , y_t ,  x_{t+1} , y_{t+1}) = \sqrt{(x_t - x_{t+1})^2 + (y_t - y_{t+1})^2)}$$
- $$S(vx_{t+1} , vy_{t+1}, k)$$: The velocity award function. Awards or punishes any acceleration
    - $$\huge S(vx_{t+1} , vy_{t+1}, k) = k\sqrt{vx_{t+1}^2 + vy_{t+1}^2}$$
      
- A(vx_{t} , vy_{t}, vx_{t+1}, vy_{t+1}): The turn penalty, punishes a change in angle (added to prevent perpetual spinning). Let v1 be the vector $$v1 = [vx_t , vy_t]^T$$ and v2 be the vector $$v2 = [vx_{t+1} , vy_{t+1}]^T$$. Then denote the euclidian norm $$<. , .>$$. We use the dot product identity to relate theta to the dot of vectors v1 and v2 and retrieve theta like so:
    - $$\huge A(vx_{t} , vy_{t}, vx_{t+1}, vy_{t+1}) = arccos(\frac{<v1,v2>}{|v1||v2|})$$

- $$P(sensor1, ..., sensork$$: returns the minimum sensor distance:
    - $$\huge P(sensor1, ..., sensork) = e^{-min(sensor1, ..., sensork)}$$
My reason for using e here was being exponential functions (with a negative coefficent) will increasingly punish the agent with a punishment closer to 1 as the agent approaches a target. This provides a layer of sensativity that can be increased by adjusting $$\nu$$ where lower values of $$\nu$$ will increasingly punish the agent. 

Constants $\alpha, \beta, \gamma, \nu$ are intended to weight each of these values accordingly. In my program, my weights are 0.1, 0.001, -1, -3 respectively. 


# Results

After about 5000 training epochs, the car could sucessfully drive on its own around the track without bumping into any walls. The car maintains a nearly fixed distance from the internal wall and travels around the track at a near constant speed (more plots on this soon). When faced with an obstruction, the car will dodge it and return back to its intended track of motion. An issue that occurs with the setup is that equidistant sensors makes it harder for the car to see well in advance since some targets slip the view of the sensors until its too late for the agent to dodge. A simple fix to this would be a larger model with more sensors or perhaps letting the sensors not be equidistant from each other and instead concentrate them at the front of the car to detect the path of motion. The car also tends to take a path that is closer to the inside of the track. A car traveling on the outside track would be more efficent at gaining distance. 


# Additional Notes
The project and data collection was done on an online environment created by Professor Young Wu, a Artifical Intelligence professor at the University of Wisconsin - Madison. You can access his environment [here](https://pages.cs.wisc.edu/~yw/index.html). Since paths may be different in the training process for different developers, reproducing the same behaviors may be difficult. I have saved the weights in the file best_network.txt, which holds the formatted weights as per the websites specification. While observing the training process, I also saved the weights of networks that led to interesting behaviors in the "manual.txt" file. 




