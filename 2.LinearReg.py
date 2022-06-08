import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./dataset/Google_Stock_Price_Train.csv')

# for i,e in enumerate(dataset['Volume']):
# 	dataset['Volume'][i]=e.replace(',', '')

# x = np.array(dataset.iloc[:,1:5])
# y= np.array(dataset.iloc[:,-1])

x = np.array(dataset.iloc[:,1:4])
y= np.array(dataset.iloc[:,-3])

x=StandardScaler().fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("Actual    : ", y_test)
print("Predicted : ", y_pred)
z=x_test.T

# plt.plot(x_train, model.coef_*x_train+model.intercept_, color="k")
plt.plot(x_train, y_train, "*", color="r")
plt.plot(z[0], y_test, "o", color="b")
plt.plot(z[0], y_pred, "x", color="g")


MSE = mean_squared_error(y_test, y_pred)
print("The MSE is : ",MSE)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("The RMSE is : ",RMSE)

MAE = mean_absolute_error(y_test, y_pred)
print("The MAE is : ",MAE)

plt.show()