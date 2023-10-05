# Import necessary libraries
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['Target'] = boston.target

### Linear Regression based on OLS

# use 70% of the date for the training
train=df[:350]
test=df[350:]

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

x_val_train=np.linspace(X_train.min(),X_train.max(),len(X_train)) 
y_predict = intercept + slope * x_val_train

# Plotting on training data
plt.scatter(X_train, Y_train, color='blue', label='Training Data')
#plt.scatter(X_test, Y_test, color='red', label='Test Data')
plt.plot(x_values, y_predict, color='black', label='Regression Line')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Target (Price)')
plt.title('Linear Regression of House Price against Number of Rooms')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

x_val_test=np.linspace(X_test.min(),X_test.max(),len(X_test))
y_predict_test = intercept + slope * x_val_test


# Plotting on test data
plt.scatter(X_test, Y_test, color='red', label='Test Data')
plt.plot(x_values_test, y_predict_test, color='black', label='Regression Line')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('Target (Price)')
plt.title('Linear Regression of House Price against Number of Rooms')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()



### calculate the error

## absolute mean error
MAE_train=abs((((Y_train-y_predict)).sum())/len(Y_train))
MAE_test=abs(((Y_test-y_predict_test).sum())/len(Y_test))
print("MAE on training data",MAE_train)
print("MAE on test data",MAE_test)

## mean squared error
MSE_train=(((Y_train-y_predict)**2).sum())/len(Y_train)
print("MSE on training data",MSE_train)
MSE_test=(((Y_test-y_predict_test)**2).sum())/len(Y_test)
print("MSE on test data",MSE_test)

## R^2

RSqure_train=1-((((Y_train-y_predict)**2).sum())/((Y_train-Y_train.mean())**2).sum())
print("RSquare on training data",RSqure_train)

RSqure_test=1-(((Y_test-y_predict_test)**2).sum())/(((Y_test-Y_test.mean())**2).sum())
print("RSquare on test data",RSqure_test)

    
    

    
    
#### Multivarialbe Regression:
