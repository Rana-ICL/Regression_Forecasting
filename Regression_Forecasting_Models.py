# Import necessary libraries
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['Target'] = boston.target

### let' split the data

mask = np.random.rand(len(df)) < 0.7
train=df[mask]
test=df[~mask]

X_train=train["RM"]
Y_train=train["Target"]

X_test=test["RM"]
Y_test=test["Target"]

X_train_mean=X_train.mean()
Y_train_mean=Y_train.mean()

X_train_array=X_train
Y_train_array=Y_train

x_var=X_train_array-X_train_mean
y_var=Y_train_array-Y_train_mean
x_var_square=(X_train_array-X_train_mean)**2
xy_var=x_var*y_var

slope=((xy_var.sum()))/(x_var_square.sum())
intercept=Y_train_mean - slope*X_train_mean

  
x_values = np.linspace(X_train.min(), X_train.max(), 100)  # Generates 100 equally spaced values from min to max
y_predict = intercept + slope * x_values  

# Plotting
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
plt.scatter(X_test, Y_test, color='red', label='Test Data')
plt.plot(x_values, y_predict, color='green', label='Regression Line')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Target (Price)')
plt.title('Linear Regression of House Price against Number of Rooms')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
