<h1><a href="https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7"> K Nearest Neighbors</a>
</h1>


from KNNClassifier import knnClassifier

X_train = features to train
y_train = labels to train
X_test = testing features
y_test = testing features


clf = knnClassifier(k=3)
clf.fit(X_train,y_train)
pred = clf.pred(X_test)
acc_knn = clf.accuracy(y_test,pred)
print(:Accuracy of the KNN Algorithm is : ",acc_knn)



#Comparing with sklearn 
from sklearn import neighbors
clf_sk_knn = neighbors.KNeighborsClassifier(n_neighbors=n)
clf_sk_knn.fit(X_train,y_train)
pred_sk_knn = clf_sk_knn.predict(X_test)
accu_sk_knn = accuracy_score(pred_sk_knn,y_test)
print("Accuracy of SKLearn KNN : ", accu_sk_knn)
