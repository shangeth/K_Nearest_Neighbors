from scipy.spatial import distance
from collections import Counter

def eucl(a,b):
    return distance.euclidean(a,b)

class knnClassifier:

    def __init__(self,k):
        self.k = k

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def k_closest(self,row):
        distances=[]
        for i in range(len(self.X_train)):
            dist = eucl(row,self.X_train[i])
            distances.append([dist,self.y_train[i]])

        k_sorted_distances = [i[1] for i in sorted(distances)[:self.k]]
        pred = Counter(k_sorted_distances).most_common(1)[0][0]

        return pred

    def predict(self,X_test):
        predictions=[]
        for row in X_test:
            label =self.k_closest(row)
            predictions.append(label)
        return predictions

    def accuracy(self):
        pass