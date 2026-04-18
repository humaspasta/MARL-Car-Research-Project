import torch
import math
import torch.nn as nn
import pandas as pd


gamma = 0.5
loss_fn = nn.MSELoss()

def calculate_reward(s, s_next):
    # example: distance reduction (last column)
    return s[-1] - s_next[-1]

def train(model, optimizer, states, epochs):
    """
    states: Tensor of shape (T, state_dim)
    """
    for epoch in range(epochs):
         
        for t in range(len(states) - 1):
            curr_s = states[t]
            next_s = states[t+1]

            output_qs = model(states[t])
      
            max_q, action = torch.max(q_values, dim=0)
            
            reward = calculate_reward(states[t])

            #calculate the reward for that state
            with torch.no_grad():
                next_q_values = model(next_s)
                max_next_q = torch.max(next_q_values)
                target_q = reward + gamma * max_next_q

            predicted_q = q_values[action]

            # Compute loss
            loss = loss(predicted_q, target_q)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss:.4f}")

                 
def calculate_reward(s):
    return (math.sqrt((s.vx)**2 + (s.vy)**2)) ** 0.5

























for epoch in range(epochs):
        total_loss = 0.0

        for t in range(len(states) - 1):

            s = states[t]
            s_next = states[t + 1]

            q_values = model(s)

            action = torch.argmax(q_values)

            q_sa = q_values[action]

            r = calculate_reward(s)

            with torch.no_grad():
                max_next_q = model(s_next).max()
                target = r + gamma * max_next_q

            # loss
            loss = loss_fn(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss = {total_loss:.4f}")

