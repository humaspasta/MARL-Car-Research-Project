import torch.nn as nn
import torch 



class DQN(torch.nn.Module):

    def __init__(self, n=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(n , 50)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(50, 25)
        self.layer3 = nn.Linear(25 , 4)
        # self.soft = nn.Softmax(dim=-1)

    def forward(self , x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x        


class PGM(torch.nn.Module):
    def __init__(self, n , *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Linear(n , 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25 , 4)
        )
    
    def forward(self , x):
        return self.layers(x)
        

class Critic(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)   # single value output
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # shape: (batch,)

                
                







