import training
import DQN 
import agent 
import torch




if __name__ == "__main__":
    website = "https://pages.cs.wisc.edu/~yw/CS540car.html"
    model = DQN.DQN()
    optim = torch.optim.Adam(params=model.parameters() , lr=1e-3)

    agent_instance = agent.Agent(website , model)


    for epoch in range(1):
        print(f'Epoch #: {epoch}')
        data = agent_instance.load_and_get_data()
        training.train(model, optim , data)
    with open("params.txt" , 'w') as file:
        file.write(agent_instance.encode_model())
    
    test_val1 = torch.tensor([ 0.0796, 0.1125, 0.2516, 0.1125, 0.0905] ,dtype=torch.float32)
    test_val2 = torch.tensor([0.0796, 0.1125, 0.2520, 0.1125, 0.0954],dtype= torch.float32)

    result = training.predict(model , test_val1)
    result1 = training.predict(model , test_val2)
    print(torch.argmax(result).detach())
    print(torch.argmax(result1).detach())

    #[nothing , speed up, turn left, turn right]


    
    