import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt

dataset = pd.read_csv("./dataset/tennis.csv")

for label in dataset.columns:
	dataset[label] = LabelEncoder().fit_transform(dataset[label])

data = np.array(dataset.drop(["play","day"], axis=1))
target = np.array(dataset["play"])

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

id3 = DecisionTreeClassifier()
id3.fit(x_train, y_train)
y_pred = id3.predict(x_test)
# plot_tree(id3)

print("Actual Output: ", y_test)
print("Predicted Output: ", y_pred)

accuracy = id3.score(x_test, y_test)
print("Accuracy : ", accuracy)

print("confusion Matrix : ")
print(confusion_matrix(y_test, y_pred))
print("Classificatiin report : ")
print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(report)
df.T.plot(kind='bar')
plt.show()

#Classify a new example
New_example=np.array(['Rain','Cool','Normal','Weak'])
encoded=LabelEncoder().fit_transform(New_example)
encoded=encoded.reshape(1,-1)
new_pred=id3.predict(encoded)
print(new_pred)
