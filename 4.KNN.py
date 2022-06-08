import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# dataset = pd.read_csv("../Datasets/Iris.csv")

# for label in dataset.columns:
#     dataset[label] = LabelEncoder().fit_transform(dataset[label])

# data = np.array(dataset.drop(["Species", "Id"], axis=1))
# target = np.array(dataset["Species"])

dataset = datasets.load_iris()
data = dataset.data
target = dataset.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Actual Output :    ", y_test)
print("Predicted Output : ", y_pred)

print("confusion Matrix : ")
print(confusion_matrix(y_test, y_pred))
print("Classificatiin report : ")
print(classification_report(y_test, y_pred))

for class_value in range(3):
    row_ix = np.where(y_pred== class_value)
    row_px = np.where(y_test== class_value)

    if(class_value==0):
        m='*'
        c='red'
    elif(class_value==1):
        m="o"
        c='green'
    elif(class_value==2):
        m='x'
        c='yellow'
    plot1 = plt.figure(1)
    plt.plot(x_test[row_ix, 1], x_test[row_ix, 0],marker=m,color=c)

    if(class_value==0):
        m='*'
        c='violet'
    elif(class_value==1):
        m="o"
        c='black'
    elif(class_value==2):
        m='x'
        c='cyan'
    plot2= plt.figure(2)
    plt.plot(x_test[row_px, 1], x_test[row_px, 0],marker=m,color=c)
plt.show()
