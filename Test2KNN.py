import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import time


# Loading the prostate cancer dataset from the csv file using pandas
data = pd.read_csv('CleanedData.csv', header=0)
#data.columns = ['IN', 'GRE', 'TOEFL', 'RATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ASIAN', 'AA', 'LATI', 'WHI', 'CHAN']
#features = ['GRE', 'TOEFL', 'RATE', 'SOP', 'LOR', 'CGPA', 'RES', 'SES', 'ASIAN', 'AA', 'LATI', 'WHI']

columns = data.columns
X = data[columns[0:-1]]
median_value = data["Chance of Admit"].median()
for i in range(len(data["Chance of Admit"])):
    if data.loc[i, "Chance of Admit"] >= 0.73:
        data.loc[i, "Chance of Admit"] = 1
    else:
        data.loc[i, "Chance of Admit"] = 0

y = data["Chance of Admit"]
print('Class labels:', np.unique(y))
#y = y.astype('int').T
print('Class labels:', np.unique(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Standardize the features:
scaler = pp.PowerTransformer()
#scaler = pp.StandardScaler()
#scaler = pp.MaxAbsScaler()
#scaler = pp.RobustScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# K-nearest neighbors
knn_k = [5, 10, 15, 20, 25, 30, 35]
dist_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']
res_pred = []
res_times = []
for i in knn_k:
    t_b = time.perf_counter()
    knn = KNeighborsClassifier(n_neighbors=i, metric='manhattan', weights='distance')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    res_pred.append(accuracy_score(y_test, y_pred))
    t_e = time.perf_counter()
    res_times.append((t_e - t_b)*1000)
plt.plot(knn_k, res_pred, 'r--', knn_k, res_times, 'b--')
plt.xlabel('K')
plt.ylabel('Accuracy (red) and Time (blue)')
plt.show()

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


# Best result given by n = 15, metric = 'manhattan', weights = 'distance'
knn_optimal = KNeighborsClassifier(n_neighbors=15, metric='manhattan', weights='distance')
knn_optimal.fit(X_train_std, y_train)
y_optimal_pred = knn_optimal.predict(X_test_std)
print("KNN Accuracy: {}".format(accuracy_score(y_test, y_optimal_pred)))
print("Metrics: {}".format(classification_report(y_test, y_optimal_pred)))


