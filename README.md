# Reinforcement Learning Car Agent
 
A reinforcement learning project that trains a car to maximize the total distance it
travels around a confined, circular track without crashing. A single agent is trained
in a **competitive, multi-car environment**: other cars share the track and collisions
(with walls or other cars) reduce the agent's speed by 25%, but the learning itself is
single-agent — one network learns to maximize its own distance. The best network found
so far is saved in `best_network.txt`.
 
## Problem setup
 
The agent competes to travel the farthest distance in a fixed number of frames.
Collisions are not penalized explicitly, but they reduce the car's speed by 25%, so
avoiding them is implicitly rewarded through the distance objective.
 
The action space is **discrete**: at each state the agent chooses to turn left, turn
right, speed up, or do nothing. "Do nothing" still produces slow forward motion, so it
is not a true no-op.
 
## Repository structure
 
| File | Purpose |
| --- | --- |
| `agent.py` | Selenium bridge to the environment: loads network weights, runs the simulation, scrapes states, and computes rewards |
| `DQN.py` | Network definitions (the shared MLP used for both DQN and policy gradient) |
| `training.py` | Training loops (policy gradient / REINFORCE and DQN) |
| `Main.py` | Entry point that runs the training loop |
| `best_network.txt` | Best network weights found, in the environment's expected format |
| `manual.txt` | Weights for networks that produced interesting behaviors during training |
 
## Tech stack
 
Python, PyTorch, Selenium, NumPy.
 
## 1. Data collection
 
Data collection and formatting are handled with **Selenium**, which drives a browser-based
simulator hosted online. Each training iteration, the agent loads the current network's
weights into the environment, runs the simulation, and scrapes the resulting sequence of
states — so it collects fresh rollouts under the current policy rather than learning from
a fixed, pre-collected dataset.
 
For each time step $t$, the environment returns a state tuple
 
$$
(a_t, x, y, v_x, v_y, \text{sensor}_1, \text{sensor}_2, \dots, \text{sensor}_n)
$$
 
where $n$ is odd. Each sensor is a ray cast radially outward from the agent that returns
the distance to the nearest object it intersects (another car or a track wall).
 
`agent.py` handles this loop: it scrapes the website, injects the network's weights in the
required format, lets the simulation run until 1,000+ steps are collected, and formats the
output into `(state, action, reward)` tuples. The reward is computed inside `agent.py`.
 
**Pipeline:** `agent.py` → website → data formatter (in `agent.py`) → formatted data.
 
## 2. The model
 
A sequential MLP, shared across both the DQN and policy gradient methods:
 
$$
\text{Linear}(n, 50) \rightarrow \text{ReLU} \rightarrow \text{Linear}(50, 25) \rightarrow \text{ReLU} \rightarrow \text{Linear}(25, 4)
$$
 
The four output values correspond to the four discrete actions.
 
## 3. The reward function
 
The reward function was designed through trial and error. The central challenge was
rewarding good behavior without *implicitly* encouraging bad behavior. Two degenerate
policies the agent discovered early on were:
 
- **Perpetual spinning** — maximizing sensor values by circling in place.
- **Speed-only** — maximizing distance while ignoring collisions.
The final reward combines four terms. Given the current state $s_t$, next state
$s_{t+1}$, and initial state $s_0$:
 
$$
R(s_t, s_{t+1}, s_0) = \alpha \, d + \beta \, S + \gamma \, A + \nu \, P
$$
 
**Displacement** $d$ — Euclidean distance moved between steps:
 
$$
d(x_t, y_t, x_{t+1}, y_{t+1}) = \sqrt{(x_t - x_{t+1})^2 + (y_t - y_{t+1})^2}
$$
 
**Speed** $S$ — magnitude of the velocity, rewarding forward motion:
 
$$
S(v_{x,t+1}, v_{y,t+1}) = \sqrt{v_{x,t+1}^2 + v_{y,t+1}^2}
$$
 
**Turn penalty** $A$ — penalizes changes in heading, added specifically to kill the
spinning policy. With $v_1 = [v_{x,t}, v_{y,t}]^\top$ and $v_2 = [v_{x,t+1}, v_{y,t+1}]^\top$,
the angle between consecutive velocity vectors is recovered from the dot-product identity:
 
$$
A(v_1, v_2) = \arccos\\left( \frac{|\langle v_1, v_2 \rangle|}{\|v_1\| \, \|v_2\|} \right)
$$
 
**Proximity penalty** $P$ — penalizes getting close to obstacles:
 
$$
P(\text{sensor}_1, \dots, \text{sensor}_k) = e^{-\min(\text{sensor}_1, \dots, \text{sensor}_k)}
$$
 
An exponential is used so the penalty rises sharply toward 1 as the nearest obstacle
approaches, giving the agent increasing sensitivity as it nears a collision. The
weight $\nu$ tunes that sensitivity.
 
The weights $\alpha, \beta, \gamma, \nu$ balance the four terms. In this project they
are $0.1,\ 0.001,\ -1,\ -3$ respectively.
 
## Results
 
After roughly 5,000 training epochs, the car drives around the track on its own without
hitting walls. It holds a nearly fixed distance from the inner wall, travels at a near-
constant speed, and — when faced with an obstacle — dodges it and returns to its path.

 
**Known limitations:**
 
- **Sensor layout.** Equidistant sensors limit foresight: some obstacles slip between
  sensor rays until it is too late to react. Concentrating more sensors toward the front
  of the car, or adding sensors overall, would improve early detection.
- **Path choice.** The agent tends to hug the inside of the track. A path along the
  outside would cover more distance per lap and be more efficient.
## Notes
 
The environment was created by Professor Young Wu, an Artificial Intelligence professor
at the University of Wisconsin–Madison; it is available
[here](https://pages.cs.wisc.edu/~yw/index.html). Because file paths differ between
setups, reproducing identical behavior can be difficult. The best weights are saved in
`best_network.txt` (formatted to the environment's specification), and weights that
produced notable behaviors during training are saved in `manual.txt`.
 
