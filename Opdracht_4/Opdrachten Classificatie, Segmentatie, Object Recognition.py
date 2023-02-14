import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

clf = svm.SVC(gamma=0.001, C=100)
X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)

print(clf.predict(digits.data[-4:-3])) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

print('test')