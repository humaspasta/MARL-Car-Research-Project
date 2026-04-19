import torch
import math
import random
import torch.nn as nn
import numpy as np


gamma = 0.1
loss_fn = nn.MSELoss()
epsilon = 0.8


def train(model, optimizer, states):
    """
    states: Tensor of shape (T, state_dim)
    """
    model.train()
    for i in range(len(states) - 1):
        curr_s = torch.tensor(states[i][1:] ,dtype=torch.float32)
        next_s = torch.tensor(states[i+1][1:], dtype=torch.float32)
        curr_action = int(states[i][0].item())
        
        curr_reward = calculate_reward(curr_s , next_s)
        q_vals = model(curr_s.unsqueeze(0)).squeeze(0)
        q_sa = q_vals[curr_action]

        with torch.no_grad():
            target = curr_reward + gamma * torch.max(model(next_s)).detach()
        loss = loss_fn(q_sa , target)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    
def calculate_reward(s, s_next):
    state1 = s[1:]
    state2 = s_next[1:]
    net_state = torch.abs(state2 - state1)
    return torch.sum(net_state)



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






