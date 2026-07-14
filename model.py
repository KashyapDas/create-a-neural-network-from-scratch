import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
# Create a class that inherit the nn.Module
class Model(nn.Module):
    # 4 input layer, in_features = 4
    # 2 Hidden layer, 1st layer has 8 neurons, 2nd layer has 9 neurons
    # 3 output layer, out_features = 3
    def __init__(self,in_features=4,h1=8,h2=9,out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,out_features)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Pick a manual seed for randomization
torch.manual_seed(41)
#Create the instance of the Model
model = Model()
# Load the dataset
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_def = pd.read_csv(url)
# Modify the dataset 
my_def['species'] = my_def['species'].replace('setosa',0.0)
my_def['species'] = my_def['species'].replace('versicolor',1.0)
my_def['species'] = my_def['species'].replace('virginica',2.0)
# Train test split! 

