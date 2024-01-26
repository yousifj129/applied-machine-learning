import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
   
# Read data from data.csv file
data = []
with open("./autoSwedishInsurance/autoSwedishInsurance.csv", "r") as file:
    for line in file:
        row = line.strip().split(",")
        data.append([float(row[0]), float(row[1])])

# Separate x and y values
x_values = [row[0] for row in data]
y_values = [row[1] for row in data]

# create dummy data for training
y_train = np.array(y_values, dtype=np.float32).reshape(-1, 1)
x_train = np.array(x_values, dtype=np.float32).reshape(-1,1)

model = LinearRegression()

model.fit(x_train, y_train)

y_pred = model.predict(x_train)

plt.clf()
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
plt.plot(x_train, y_pred, '--', label='Predictions', alpha=0.5)
plt.legend(loc='best')
plt.show()