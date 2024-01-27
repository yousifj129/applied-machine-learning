import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Step 1: Preprocess the Data
data = pd.read_csv("./banknoteAuth/data_banknote_authentication.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the Neural Network Model
def get_training_model(inFeatures=5, hiddenDim=128, nbClasses=1):
    mlpModel = nn.Sequential(
        nn.Linear(inFeatures, hiddenDim),
        nn.ReLU(),
        nn.Linear(hiddenDim, nbClasses),
        nn.Sigmoid()
    )
    return mlpModel

model = get_training_model(4, 1000, 1)

# Step 3: Train the Model
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10000
for epoch in range(num_epochs):
    inputs = torch.Tensor(X_train)
    labels = torch.Tensor(y_train)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Step 4: Evaluate the Model
with torch.no_grad():
    inputs = torch.Tensor(X_test)
    predicted_probs = model(inputs).squeeze().numpy()
    predicted_labels = (predicted_probs >= 0.5).astype(int)
    
    accuracy = (predicted_labels == y_test).mean()
    print(f"Accuracy: {accuracy}")

torch.save(model.state_dict(), "./banknoteAuth/model.pth")
print("Model saved successfully.")