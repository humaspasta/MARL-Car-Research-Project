from copy import deepcopy
from DQN import Critic
from agent import Agent
import torch
import math
import random
import torch.nn as nn
import numpy as np


gamma = 0.9
loss_fn = nn.MSELoss()


# def train(model, optimizer, states):
#     """
#     states: Tensor of shape (T, state_dim)
#     """
#     model.train()
#     for i in range(len(states) - 1):
#         curr_s = torch.tensor(states[i][1:] ,dtype=torch.float32)
#         next_s = torch.tensor(states[i+1][1:], dtype=torch.float32)
#         curr_action = int(states[i][0].item())
        
#         curr_reward = calculate_reward(curr_s)
#         q_vals = model(curr_s.unsqueeze(0)).squeeze(0)
#         q_sa = q_vals[curr_action]

#         with torch.no_grad():
#             target = curr_reward + gamma * torch.max(model(next_s)).detach()

#             print("current vs target")
#             print(q_sa , target)
#         loss = loss_fn(q_sa , target)
#         print(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(loss.item())

def train(model, optimizer, states, global_step=0):
    old_model = deepcopy(model)
    old_model.eval()

    model.train() 
    total_loss = 0.0
    epsilon = 0.8
    for i in range(len(states) - 1):
        if global_step % 100 == 0:
            old_model.load_state_dict(model.state_dict())

        curr_s = torch.tensor(states[i][1:], dtype=torch.float32)
        next_s = torch.tensor(states[i+1][1:], dtype=torch.float32)
        
        full_s_data = torch.tensor(states[i], dtype=torch.float32)
        full_next_s_data = torch.tensor(states[i+1], dtype=torch.float32)

        q_vals = model(curr_s[4:].unsqueeze(0)).squeeze(0)

        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = torch.argmax(q_vals).item()
        print(f"Q-values: {q_vals.detach().numpy()}")
        q_sa = q_vals[action]   

        with torch.no_grad():
            next_q_vals = old_model(next_s[4:].unsqueeze(0)).squeeze(0)
            max_next_q = torch.max(next_q_vals).item()
            reward = calculate_reward(full_s_data , full_next_s_data)
            print("Reward:", reward.item())
            target = reward + gamma * max_next_q

        loss = loss_fn(q_sa, target) * 10

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        epsilon = max(0.1, epsilon * 0.8)

    avg_loss = total_loss / (len(states) - 1)
    print(f"Avg Loss: {avg_loss:.4f}")


def train_REINFORCE(model , agent:Agent, optimizer):
    data = agent.load_and_get_data(samples=1000 , frame=1)
    loss = 0
    for i , state in enumerate(data):
        state_data = torch.tensor(state[0][5:] ,dtype=torch.float32).unsqueeze(0)
        gamma = 0.9
        action = torch.tensor(state[1] , dtype=torch.int64)

        #computing the returns
        returns = 0

        for j in range(i , len(data)):
            
            reward = data[j][2]
            returns += math.pow(gamma , j -i ) * reward
        
        #updating the model 
        logits = model(state_data)
        if action == -1:
            continue
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        loss += -returns * log_prob
    
    optimizer.zero_grad()
    loss = loss /len(data)
    print(loss)
    loss.backward()
    optimizer.step()













def train_better(model, actor, optimizer, batch , global_step = 0):
    model.train()
    actor.train()
    total_loss = 0.0
    vals = 1

    rewards = []
    log_probs = []
    value_estimates = []

    for (s, a, r) in batch:
        action = torch.tensor(a , dtype=torch.int64)
        if action == -1:
            continue
        
        state = torch.tensor(s, dtype=torch.float32).unsqueeze(0)

        reward = torch.tensor(r, dtype=torch.float32)
        value = actor(state[: , 5:]).squeeze()

        logits = model(state[: , 5:])

        dist = torch.distributions.Categorical(logits=logits)
        
        log_prob = dist.log_prob(action)
        # print("The log probability is what " + str(log_prob))
    
        log_probs.append(log_prob)
        rewards.append(reward)
        value_estimates.append(value)


    returns = []
    G = 0  
    value_estimates = torch.stack(value_estimates).squeeze()
    for i in reversed(range(len(rewards))):
        if batch[i][0][0] == -1:  # new episode marker
            G = 0
        G = rewards[i] + gamma * G
    
        returns.insert(0, G)

    returns = torch.stack(returns)

    returns_std = returns.std(unbiased=False)
    if returns_std > 1e-8:
        returns = (returns - returns.mean()) / (returns_std + 1e-8)
    else:
        returns = returns - returns.mean()

    log_probs = torch.stack(log_probs)
    advantage = returns - value_estimates.detach()
    loss = -(log_probs * advantage).mean()

    critic_loss = (value_estimates - returns).pow(2).mean()

    loss = loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()


    print(f"Avg Loss: {total_loss / len(batch):.4f}")

    
def calculate_reward(s, s_next , batch):
    crash = (s_next[0] == -1)

    if crash:
        return -10.0
    
    angle_change = torch.arccos(torch.dot(s_next[3:5], s[3:5]) / (torch.norm(s_next[3:5]) * torch.norm(s[3:5]) + 1e-8))
    if angle_change > 1:
        print(angle_change)
        return -5.0
    forward_progress = s_next[1]  - s[1] + s_next[2] - s[2]
    sum = torch.sum(s_next[4:]) - torch.sum(s[4:])
    reward = forward_progress * 10 + sum * 0.1
    return torch.clamp(reward, -1.0, 1.0) 

#invalid = -1
#turn right = 0
# turn left = 1
# speed up = 2
# do nothing = 3


def predict(model, state):
    model.eval()
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32)
        return model(state.unsqueeze(0)).squeeze(0)

def manual_predict(layers, state):
    x = np.array(state)

    for i, (W, b) in enumerate(layers):
        x = np.dot(W, x) + b
        if i < len(layers) - 1:
            x = np.maximum(0, x)  # ReLU

    return x





def train_es(model, agent):
    sigma = 0.02
    alpha = 0.01
    theta = get_params(model)

    rewards = []
    noises = []
    population_size = 10

    for i in range(population_size):
        print(i)
        noise = torch.randn_like(theta)

        theta_try = theta + sigma * noise
        set_params(model, theta_try)
        data = agent.load_and_get_data()

        start_val_x = data[0][0][1]
        start_val_y = data[0][0][2]

        last_val_x = data[-1][0][1]
        last_val_y = data[-1][0][1]

        reward = 0

        if not check_crash(data):
            reward = math.sqrt((last_val_x - start_val_x) + (last_val_y - start_val_y))
        

        # reward = sum([r for (_, _, r) in data])
        print(reward)
        reward = torch.tensor(reward , dtype=torch.float64)

        rewards.append(reward)
        noises.append(noise)

    rewards = torch.tensor(rewards)
    # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)


    update = torch.zeros_like(theta)

    for r, n in zip(rewards, noises):
        update += r * n

    theta = theta + alpha * update / population_size

    set_params(model, theta)

    print("Avg reward:", rewards.mean().item())

def get_params(model):
    return torch.cat([p.data.flatten() for p in model.parameters()])

def check_crash(data):
    for dat in range(len(data)):
        if data[dat][0][0] == -1:
            return True
    return False
def set_params(model, flat_params):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx:idx+numel].view_as(p))
        idx += numel