from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import torch
import numpy as np
import time 


class Agent:
    def __init__(self , website , model:torch.nn.Module):
        self.website = website
        self.options = Options()
        self.options.add_argument("--headless=new")
        self.driver = webdriver.Chrome(options=self.options)
        self.model = model

    def open_website(self):
        self.driver.get(self.website)
        time.sleep(3) 

    def get_model_params(self):
        return self.model.paramters()

    def load_and_get_data(self):
        self.open_website() #opening the website
        time.sleep(5)
        data_element = self.driver.find_element(By.ID , "net") #getting the textarea element where I need to input the network
        params = self.encode_model()
        if data_element and data_element.tag_name == "textarea":
            print("...obtained text area...")
            print("...sending network data to the text area...")
            data_element.send_keys(params)
        else:
            print("Data element not found or is not a textarea.")
            raise Exception("Data element not found or is not a textarea.")

        out_limit = self.driver.find_element(By.ID , "out_limit")
        if out_limit and out_limit.tag_name == "input":
            print("...obtained output limit input...")
            out_limit.clear()
            out_limit.send_keys("10000")
        else:
            print("Output limit element not found or is not an input.")
            raise Exception("Output limit element not found or is not an input.")
        
        frame_limit = self.driver.find_element(By.ID , "out_frame")
        frame_limit.clear()
        frame_limit.send_keys("1")

        button = self.driver.find_element(By.ID , "button_0") #getting the button element to start the simmulation

        if button and button.tag_name == "button":
            print("...starting data collection...")
            button.click()
        else:
            print("Button element not found or is not a button.")
            raise Exception("Button element not found or is not a button.")
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
        formatted_data = self.format_data(data)

        return np.array(formatted_data)
    

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
    
    def encode_model(self):
        parts = []
        parts.append(self.format_linear(self.model.layer1))
        parts.append("-----")

        parts.append(self.format_linear(self.model.layer2))
        parts.append("-----")

        parts.append(self.format_linear(self.model.layer3))

        return "\n".join(parts)
    
    def format_data(self , data:str):
        formatted_data_tules = []
        data = data.split("\n")
        for line in data:
            if line.strip():
                values = line.split(",")
                try:
                    values = [float(value) for value in values]
                    formatted_data_tules.append(tuple(values))
                except ValueError:
                    print(f"Warning: Could not convert line to floats: {line}")
        return formatted_data_tules
       

    def close(self):
        self.driver.quit()

    

    
#1. Data is not as stated on website [1:] is sensor data
#2. train on more data 
#3. reward function is incorrect and need new way to calculate velocity