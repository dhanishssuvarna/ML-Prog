import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# Load dataset
data = pd.read_csv("../Datasets/Salary_Data.csv")

# seperating data
x = np.array(data.iloc[:, 0:1])
y = np.array(data.iloc[:, -1])

x = StandardScaler().fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


from LR import linear_reg
model = linear_reg()

# model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Actual    : ", y_test)
print("Predicted : ", y_pred)
z = X_test.T

plt.plot(X_train,model.w*X_train+model.b,color='g')
# plt.plot(X_train, model.coef_ * X_train + model.intercept_, color="k")
plt.plot(X_train, y_train, "*", color="r")
plt.plot(z[0], y_test, "o", color="b")
plt.plot(z[0], y_pred, "o", color="g")

MSE = mean_squared_error(y_test, y_pred)
print("the MSE is:", MSE)

RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print("the root mean sq error is:", RMSE)

MAE = mean_absolute_error(y_test, y_pred)
print("the mean avg error is:", MAE)

plt.show()
