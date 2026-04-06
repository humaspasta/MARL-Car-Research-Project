import torch.nn as nn
import torch 



class DQN(torch.nn.Module):

    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(n , n/2)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(n/2 , n/4)
        self.layer3 = nn.Linear(n/4 , 4)


    
    def forward(self , x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)




