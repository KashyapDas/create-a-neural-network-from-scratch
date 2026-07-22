import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import optuna
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loading the train and test (dataset)
train_dataset = pd.read_csv("fashion-mnist_train.csv")
test_dataset = pd.read_csv("fashion-mnist_test.csv")

#training data
x_train = train_dataset.drop(columns=['label'])
y_train = train_dataset['label']
#testing data
x_test = test_dataset.drop(columns=['label'])
y_test = test_dataset['label']

# scaling the train data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Check if they are in numpy/pandas and if then convert them to the tensor
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

x_train_tensor = torch.from_numpy(x_train).float()
x_test_tensor = torch.from_numpy(x_test).float()

y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Defining the CustomClass
class customDataset(Dataset):
  def __init__(self, features, labels):
    self.features = features
    self.labels = labels

  def __len__(self):
    return self.features.shape[0]

  def __getitem__(self, index):
    return self.features[index], self.labels[index]

xy_train_dataset = customDataset(x_train_tensor, y_train_tensor)
xy_test_dataset = customDataset(x_test_tensor, y_test_tensor)

# defining the model
class Model(nn.Module):
  def __init__(self, input_dimension, output_dimension, no_hidden_layers, neuron_per_layer, dropout_rate):
    super().__init__()
    layer = []
    for i in range(no_hidden_layers):
      layer.append(nn.Linear(input_dimension, neuron_per_layer))
      layer.append(nn.BatchNorm1d(neuron_per_layer))
      layer.append(nn.ReLU())
      layer.append(nn.Dropout(dropout_rate))
      input_dimension = neuron_per_layer
    layer.append(nn.Linear(neuron_per_layer, output_dimension))
    self.model = nn.Sequential(*layer)

  def forward(self,x):
    return self.model(x)

# defining the objective for the optuna inside which the training/testing loop is defined
def objective(trial):
  print(f"Trial : {trial.number}")
  # defining the range for all the important parameters of the model
  no_hidden_layers = trial.suggest_int("no_hidden_layers", 1, 5)
  neuron_per_layer = trial.suggest_int("neuron_per_layer", 8, 128, step=8)
  epochs = trial.suggest_int("epochs", 10, 50, step=10)
  learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
  dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
  batch_size = trial.suggest_categorical("batch_size",[16,32,64,128])
  optimizer = trial.suggest_categorical("optimizer",['Adam','SGD','RMSprop'])
  weight_decay = trial.suggest_float("weight_decay",1e-5, 1e-3, log=True)

  # Defining the DataLoader Class
  train_dataloader = DataLoader(xy_train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
  test_dataloader = DataLoader(xy_test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
  # input and output dimension
  input_dimension = x_train_tensor.shape[1]
  output_dimension = len(torch.unique(y_train_tensor))
  # defining the model
  model = Model(input_dimension, output_dimension, no_hidden_layers, neuron_per_layer, dropout_rate)
  model = model.to(device)
  # defining the loss function
  criterion = nn.CrossEntropyLoss()
  # defining the optimizer
  if optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  elif optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
  else:
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  #set the model in the training mode
  model.train()
  # do the training loop
  for epoch in range(epochs):
    for batch_features, batch_labels in train_dataloader:
      batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

      prediction = model.forward(batch_features)
      loss = criterion(prediction, batch_labels)
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
    # print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


  # set the model to the testing mode
  model.eval()
  # do the evaluation/testing model
  total = correct = 0
  with torch.no_grad():
    for batch_features, batch_labels in test_dataloader:
      batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

      output = model(batch_features)
      _, predicted = torch.max(output,1)
      total = total + batch_labels.shape[0]
      correct = correct + (predicted == batch_labels).sum().item()
    accuracy = correct/total
    print(f"Accuracy is : {accuracy}")
  return accuracy

# create a study using optuna
study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials=10)


# --- ADD THIS AFTER YOUR OPTUNA STUDY ---

print("\n--- OPTUNA SEARCH FINISHED ---")
print("Best Trial:", study.best_trial.number)
print("Best Accuracy:", study.best_trial.value)
print("Best Parameters:", study.best_params)

# 1. Extract the best hyperparameters
best_params = study.best_params

# 2. Re-create the DataLoader with the best batch size
final_train_dataloader = DataLoader(
    xy_train_dataset, 
    batch_size=best_params["batch_size"], 
    shuffle=True, 
    pin_memory=True
)

# 3. Re-initialize the Model with the best architecture parameters
input_dimension = x_train_tensor.shape[1]
output_dimension = len(torch.unique(y_train_tensor))

final_model = Model(
    input_dimension=input_dimension,
    output_dimension=output_dimension,
    no_hidden_layers=best_params["no_hidden_layers"],
    neuron_per_layer=best_params["neuron_per_layer"],
    dropout_rate=best_params["dropout_rate"]
)
final_model = final_model.to(device)

# 4. Re-initialize the Loss and best Optimizer
criterion = nn.CrossEntropyLoss()

if best_params["optimizer"] == 'Adam':
    final_optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
elif best_params["optimizer"] == 'SGD':
    final_optimizer = optim.SGD(final_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
else:
    final_optimizer = optim.RMSprop(final_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])

# 5. Train the final model using the best number of epochs
print(f"\nTraining final model for {best_params['epochs']} epochs...")
final_model.train()

for epoch in range(best_params["epochs"]):
    running_loss = 0.0
    for batch_features, batch_labels in final_train_dataloader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        # Forward pass
        prediction = final_model(batch_features)
        loss = criterion(prediction, batch_labels)
        
        # Backward pass
        final_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
        final_optimizer.step()
        
        running_loss += loss.item()
        
    # Print average loss for the epoch so you can watch it train
    print(f"Epoch {epoch+1}/{best_params['epochs']} - Loss: {running_loss/len(final_train_dataloader):.4f}")

# 6. Save the trained weights to a file so others can use it
torch.save(final_model.state_dict(), "model.pt")
print("\nModel successfully saved as 'model.pt'!")