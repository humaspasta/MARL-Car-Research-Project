import random
from DQN import Critic
import training
from DQN import DQN 
import agent 
import torch
import os




if __name__ == "__main__":
    website = "https://pages.cs.wisc.edu/~yw/CS540car.html"
    model = DQN(n=7)
    critic = Critic(15)
    optim = torch.optim.Adam(params=model.parameters() , lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()
    agent_instance = agent.Agent(website , model)


    for epoch in range(1000):
        print(f'Epoch #: {epoch}')

        training.train_REINFORCE(model , agent=agent_instance ,optimizer=optim)
    agent_instance.encode_all_layers()
    
    # data = agent_instance.get_manual('./manual.txt')
    # training.train(model , optim , data)
    # with open("params.txt" , 'w') as file:
    #     file.write(agent_instance.encode_model(model))
    # test_val1 = torch.tensor([ 0.0796, 0.1125, 0.2516, 0.1125, 0.0905] ,dtype=torch.float32)
    # test_val2 = torch.tensor([0.0796, 0.1125, 0.2520, 0.1125, 0.0954],dtype= torch.float32)

    # result = training.predict(model , test_val1)
    # result1 = training.predict(model , test_val2)
    # print(torch.argmax(result).detach())
    # print(torch.argmax(result1).detach())

    #[nothing , speed up, turn left, turn right]


    
    


    