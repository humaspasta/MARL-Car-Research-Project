from selenium import webdriver
from selenium.webdriver.common.by import By
import torch
import time 


class Agent:
    def __init__(self , website):
        self.website = website
        self.driver = webdriver.Chrome()

    def open_website(self):
        self.driver.get(self.website)
        time.sleep(5) 

    def load_and_get_data(self, model: torch.nn.Module):
        self.open_website() #opening the website

        data_element = self.driver.get_element(By.ID , "net") #getting the textarea element where I need to input the network
        network = model.state_dict() #getting the state of the model 
        if data_element and data_element.tag_name == "textarea":
            data_element.send_keys(str(network))
        else:
            print("Data element not found or is not a textarea.")
            raise Exception("Data element not found or is not a textarea.")

        out_limit = self.driver.get_element(By.ID , "out_limit")
        if out_limit and out_limit.tag_name == "input":
            out_limit.clear()
            out_limit.send_keys("100")
        else:
            print("Output limit element not found or is not an input.")
            raise Exception("Output limit element not found or is not an input.")
        
        button = self.driver.get_element(By.ID , "button_0") #getting the button element to start the simmulation

        if button and button.tag_name == "button":
            button.click()
        else:
            print("Button element not found or is not a button.")
            raise Exception("Button element not found or is not a button.")
        time.sleep(30)
        data = None
        out_data_box = self.driver.get_element(By.ID , "out")
        if out_data_box and out_data_box.tag_name == "textarea":
            data = out_data_box.get_attribute("value")
        else:
            print("Output data box element not found or is not a textarea.")
            raise Exception("Output data box element not found or is not a textarea.")
        self.driver.quit()
        return data
       


    

    
            