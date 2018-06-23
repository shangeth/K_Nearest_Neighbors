import random
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


n = int(input("ENter the value of k : "))


from KNNClassifier import knnClassifier
clf_knn = knnClassifier(k=n)
clf_knn.fit(X_train,y_train)
pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test,pred_knn)
print("Accuracy of KNNClassifier : ", acc_knn)





from sklearn import neighbors
clf_sk_knn = neighbors.KNeighborsClassifier(n_neighbors=n)
clf_sk_knn.fit(X_train,y_train)
pred_sk_knn = clf_sk_knn.predict(X_test)
accu_sk_knn = accuracy_score(pred_sk_knn,y_test)
print("Accuracy of SKLearn KNN : ", accu_sk_knn)