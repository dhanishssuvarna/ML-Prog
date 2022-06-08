from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

dataset = datasets.load_breast_cancer()

data = dataset.data
target = dataset.target

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)

print("Actual Output :    ", y_test)
print("Predicted Output : ", y_pred)

print("COnfusion Matrix : ")
print(confusion_matrix(y_test, y_pred))
print("Classification Report : ")
print(classification_report(y_test, y_pred))
