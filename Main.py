import training
import DQN 
import agent 




if __name__ == "__main__":
    website = "http://localhost:8000/"
    model = DQN.DQN(100)
    agent_instance = agent.Agent(website)
    data = agent_instance.load_and_get_data(model)
    print(data)