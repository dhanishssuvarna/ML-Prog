import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn import datasets

# dataset = pd.read_csv("../Datasets/Iris.csv")
# dataset = dataset[:100]

# for label in dataset.columns:
#     dataset[label] = LabelEncoder().fit_transform(dataset[label])


# data = np.array(dataset.loc[:, ["SepalLengthCm", "SepalWidthCm"]])
# # data = np.array(dataset.loc[:, ["PetalLengthCm", "PetalWidthCm"]])
# target = np.array(dataset["Species"])

data, target = datasets.make_blobs(n_samples=100, n_features=2, centers=2, random_state=40)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

# from SVM import SVM
# model = SVM()

model = SVC(kernel="linear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Actual Output :    ", y_test)
print("Predicted Output : ", y_pred)

print("confusion Matrix : ")
print(confusion_matrix(y_test, y_pred))
print("Classificatiin report : ")
print(classification_report(y_test, y_pred))

# get the separating hyperplane
w = model.coef_[0]
m = -w[0] / w[1]
min0 = min(x_train[:, 0])
max0 = max(x_train[:, 0])
min1 = min(x_train[:, 1])
max1 = max(x_train[:, 1])
xx = np.linspace(min(min0, min1), max(max0, max1))
yy = m * xx - (model.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
up = model.support_vectors_[0]
yy_up = m * xx + (up[1] - m * up[0])
down = model.support_vectors_[-1]
yy_down = m * xx + (down[1] - m * down[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, "y-")
plt.plot(xx, yy_up, "k--")
plt.plot(xx, yy_down, "k--")

plt.xlabel("SepalLength (Cm)")
plt.ylabel("SepalWidth (Cm)")

plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap=plt.cm.Paired)

plt.show()
