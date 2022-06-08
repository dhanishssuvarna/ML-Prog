import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from KNN import k_nearest_neighbors

dataset = pd.read_csv("../Datasets/diabetes.csv")

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit_transform(dataset[label])

data = np.array(dataset.drop(["Outcome"], axis=1))
target = np.array(dataset["Outcome"])

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

knn = k_nearest_neighbors(k=3)
knn.knn_fit(x_train, y_train)
y_pred = knn.knn_predict(x_test)

print("Actual Output :    ", y_test)
print("Predicted Output : ", y_pred)

print("COnfusion Matrix : ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report : ")
print(classification_report(y_test, y_pred))
