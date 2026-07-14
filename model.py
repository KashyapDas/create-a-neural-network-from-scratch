import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
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
torch.manual_seed(32)
#Create the instance of the Model
model = Model()
# Load the dataset
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_def = pd.read_csv(url)
# Modify the dataset 
my_def['species'] = my_def['species'].map({
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
})
# Train test split -> 'x' for input features and 'y' for outcomes 
x = my_def.drop('species',axis=1)
y = my_def['species']
# Convert this to the numpy array
x = x.values
y = y.values
# Train and validation split -> Training(80%) and Validation(20%)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=32)
x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
#Set the criterion to measure the model
criterion = nn.CrossEntropyLoss()
# Choose and optimizer(adam) and set the learning the rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# print(model.parameters)
# Train our model
# Epochs ? (one run through all the training data in our neural n/w)
epochs = 100
losses = []
for i in range(epochs):
    # Go_forward and get the perdiction
    y_pred = model.forward(x_train)
    # Measure the loss/error
    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    # Do the backpropagation to fine tunning the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
# Evaluate/Validate Model on test data set
with torch.no_grad():
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
correct = 0
with torch.no_grad():
    for i,data in enumerate(x_test):
        y_val = model.forward(data)
        if y_val.argmax().item() == y_test[i]:
            correct+=1
print(f"We got {correct} correct")
#Evaluate new data on the model
index_to_species = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}
#first create a manual new data points
new_iris = torch.tensor([4.7,3.2,1.3,6.2])
# feed it to the model
with torch.no_grad():
    raw_output = model(new_iris)
    # find the predicted index
    predicted_index = raw_output.argmax().item()
    # Map the index back to the species name
    predicted_species = index_to_species[predicted_index]
    print(f"The predicted species is: {predicted_species}")
# Save the model
torch.save(model.state_dict(),'iris_model.pt')
# Load the model and evaluate it
new_model = Model()
new_model.load_state_dict(torch.load('iris_model.pt'))
print(new_model.eval())
