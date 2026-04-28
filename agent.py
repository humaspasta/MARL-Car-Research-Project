from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import torch
import numpy as np
import time 
import math


class Agent:
    def __init__(self , website , model:torch.nn.Module):
        self.website = website
        self.other_website = "https://pages.cs.wisc.edu/~yw/CS540W26CP2.html"
        self.options = Options()
        # self.options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=self.options)
        self.model = model
        self.first_state = None

    def open_website(self):
        self.driver.get(self.website)
        time.sleep(3) 

    def get_model_params(self):
        return self.model.paramters()
    
    # def load_apply_get(self , actions_list):
    #     self.driver.get(self.other_website)
    #     time.sleep(3)
    #     data_element = self.driver.find_element(By.ID , "sensor")
        
        

    def load_and_get_data(self , samples , frame):
        self.open_website() #opening the website
        time.sleep(5)
        data_element = self.driver.find_element(By.ID , "net") #getting the textarea element where I need to input the network
        self.encode_all_layers()
        if data_element and data_element.tag_name == "textarea":
            print("...obtained text area...")
            print("...sending network data to the text area...")
            with open('params.txt' , 'r') as file:
                data_element.clear()
                data_element.send_keys(file.read())
        else:
            print("Data element not found or is not a textarea.")
            raise Exception("Data element not found or is not a textarea.")

        out_limit = self.driver.find_element(By.ID , "out_limit")
        if out_limit and out_limit.tag_name == "input":
            print("...obtained output limit input...")
            out_limit.clear()
            out_limit.send_keys(f"{samples}")
        else:
            print("Output limit element not found or is not an input.")
            raise Exception("Output limit element not found or is not an input.")
        
        frame_limit = self.driver.find_element(By.ID , "out_frame")
        frame_limit.clear()
        frame_limit.send_keys(f"{frame}")

        button = self.driver.find_element(By.ID, "button_0")

        self.driver.execute_script(
        "arguments[0].scrollIntoView({block: 'center'});",
        button
        )

        time.sleep(0.5)

        print("...starting simulation...")
        self.driver.execute_script("arguments[0].click();", button)
    
        print("...30 seconds for data collection...")
        out_data_box = self.driver.find_element(By.ID , "out")
        data = None
        while not out_data_box.get_attribute("value").strip():
            continue

        data = out_data_box.get_attribute("value")
        if out_data_box and out_data_box.tag_name == "textarea":
            data = out_data_box.get_attribute("value")
        else:
            print("Output data box element not found or is not a textarea.")
            raise Exception("Output data box element not found or is not a textarea.")
        print(type(data))
        self.first_state = data[0]
        formatted_data = self.format_data(data)

        return formatted_data
    

    def extract_layer(self, matrix):
            return matrix.detach().cpu().numpy()
        
    def format_linear(self , layer: torch.nn.Linear):
        W = layer.weight.detach().cpu().numpy()
        b = layer.bias.detach().cpu().numpy()

        lines = []
        for row in W:
            lines.append(", ".join(map(str, row)))

        # biases
        lines.append(", ".join(map(str, b)))

        return "\n".join(lines)
    

    def calculate_reward(self, s, s_next, first_state):
        reward = 0
        if s_next[0] == -1:
            return -1.0
        reward += 0.3 #for not crashing
    
        prev_x, prev_y = s[1], s[2]
        curr_x, curr_y = s_next[1], s_next[2]

        first_x, first_y = first_state[1], first_state[2]

        prev_dist = math.sqrt((prev_x - first_x)**2 + (prev_y - first_y)**2)
        curr_dist = math.sqrt((curr_x - first_x)**2 + (curr_y - first_y)**2)
        displacement_reward = curr_dist - prev_dist
      

        min_sensor = min(s_next[5:])
        sensor_penalty = 1*math.exp(-min_sensor)
        speed = math.sqrt(s_next[3]**2 + s_next[4]**2)
        speed_penalty = 0.05 * speed
        dot = s_next[3] * s[3] + s_next[4] * s[4]
        norm = (math.sqrt(s_next[3]**2 + s_next[4]**2) *
                math.sqrt(s[3]**2 + s[4]**2) + 1e-8)

        angle_change = math.acos(max(-1.0, min(1.0, dot / norm)))
        turn_penalty = 0.1 * angle_change

        reward = (
            0.1 * displacement_reward
            + 0.001 * speed_penalty
            - turn_penalty - 3 * sensor_penalty
        )
        return reward

    def format_data(self , data:str):
        formatted_data_tules = []
        rl_tuples = []
        data = data.split("\n")

        for line in data:
            if line.strip():
                values = line.split(",")
                values = [float(value) if value != 'undefined' else -1.0 for value in values]
                
                values = [float(value) if value != 'undefined' else -1.0 for value in values]
                formatted_data_tules.append(tuple(values))

        # for point in range(len(formatted_data_tules) - 1):
        #     assert len(formatted_data_tules[point]) == len(formatted_data_tules[point + 1])
        #     print(formatted_data_tules[point])
        for point in range(len(formatted_data_tules) - 1):
            s_0 , a , r = formatted_data_tules[point] , formatted_data_tules[point][0] , self.calculate_reward(formatted_data_tules[point], formatted_data_tules[point + 1] , formatted_data_tules[0])
            rl_tuples.append((s_0 , a , r))

        return rl_tuples

    def get_manual(self, path):
        data = []
        with open(path , "r") as file:
            for line in file:
                print(line)
                line = line.split(",")
                line = [float(x) for x in line]
                data.append(line)
        
        return np.array(data)
    
    # def encode_layer(self, layer:torch.nn.Linear):
    #     layer_str = ""
    #     lines = []

    #     weights = layer.weight.detach().numpy()
    #     biases = layer.bias.detach().numpy()


    #     for neuron in weights:
    #         row = ",".join(f"{w:.8f}" for w in neuron)
    #         lines.append(row)

    #     bias_str = ",".join(f'{b:.8f}'for b in biases)
    #     lines.append(bias_str)
    #     return lines

    # def encode_all_layers(self):
    #     layer1 = self.model.layer1
    #     layer2 = self.model.layer2
    #     layer3 = self.model.layer3
    #     layers_list = [layer1, layer2, layer3]
    #     all_lines = []
    #     for layer in layers_list:
    #         res_str = self.encode_layer(layer)
    #         all_lines.append(res_str)
    #     print(all_lines[0])
    #     with open("params.txt" , 'w') as file:
    #         for layer_lines in all_lines:
    #             for line in range(len(layer_lines) - 1):
                
    #                 for elem in layer_lines[line]:
    #                     file.write(str(elem))
    #                 file.write("\n")

    #             bias_line = layer_lines[-1]
    #             for elem in bias_line:
    #                 file.write(str(elem))
    #             file.write("\n")
    #             file.write("-----\n")

    def encode_layer(self, layer: torch.nn.Linear, noise_scale=0.0):
        weights = layer.weight.detach().numpy().T
        biases = layer.bias.detach().numpy()

        if noise_scale > 0:
            weights = weights + noise_scale * np.random.randn(*weights.shape)
            biases = biases + noise_scale * np.random.randn(*biases.shape)

        lines = []
        
        for neuron in weights:
            row = ", ".join(f"{w:.8f}" for w in neuron)
            lines.append(row)

        bias_row = ", ".join(f"{b:.8f}" for b in biases)
        lines.append(bias_row)
        return lines
    
    def encode_all_layers(self):
        layers = [
            self.encode_layer(self.model.layer1 , noise_scale=0.00),
            self.encode_layer(self.model.layer2, noise_scale=0.00),
            self.encode_layer(self.model.layer3, noise_scale=0.0),
        ]

        with open("params.txt", "w") as f:
            for i, layer_lines in enumerate(layers):
                for line in layer_lines:
                    f.write(line + "\n")

                if i != len(layers) - 1:
                    f.write("-----\n")

        
    
    def close(self):
        self.driver.quit()

    

    
#1. Data is not as stated on website [1:] is sensor data
#2. train on more data 
#3. reward function is incorrect and need new way to calculate velocity