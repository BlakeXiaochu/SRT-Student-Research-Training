from sklearn import datasets, svm
import numpy as np

#产生二分类样本
iris = datasets.load_iris()
data, target = iris.data, iris.target
samples = data[target < 2, :]
classes = target[target < 2]

#训练SVM
clf = svm.SVC(C = 1.5, kernel = 'rbf', gamma = 'auto', decision_function_shape = 'ovr', verbose = True)
clf.fit(samples, classes)

accuracy = clf.score(samples, classes)
print('\naccuracy =', accuracy)