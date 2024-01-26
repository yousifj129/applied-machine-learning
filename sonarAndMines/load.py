import torch
import torch.nn as nn
import pandas as pd

# Define the Neural Network Model
def get_training_model(inFeatures=60, hiddenDim=128, nbClasses=1):
    mlpModel = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim),
        nn.ReLU(),
        nn.Linear(hiddenDim, nbClasses),
        nn.Sigmoid()
    )
    return mlpModel

# Load the saved model
model = get_training_model(60, 1000, 1)
model.load_state_dict(torch.load("./sonarAndMines/model.pth"))
model.eval()

# Load the data for prediction
data = pd.read_csv("./sonarAndMines/sonar-mines.csv", header=None)
X = data.iloc[:, :-1].values

# Perform predictions
with torch.no_grad():
    inputs = torch.Tensor(X)
    predicted_probs = model(inputs).squeeze().numpy()
    predicted_labels = (predicted_probs >= 0.5).astype(int)

# Print the predictions
for i, label in enumerate(predicted_labels):
    print(f"Data sample {i+1}: Predicted label = {'M' if label == 1 else 'R'}")