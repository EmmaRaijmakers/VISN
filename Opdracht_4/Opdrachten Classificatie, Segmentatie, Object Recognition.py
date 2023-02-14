import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.utils import shuffle

digits = datasets.load_digits()
print(digits.data)
print(len(digits.data))
print(digits.target)
print(len(digits.target))

#randomize data and target in the same way, to still ensure the match on the indexes
ran_data, ran_target = shuffle(digits.data, digits.target, random_state=0)

clf = svm.SVC(gamma=0.001, C=100)
# X,y = digits.data[:-10], digits.target[:-10]
X,y = ran_data[:-(len(ran_data)//3)], ran_target[:-(len(ran_target)//3)]
# X,y = digits.data[:-len(digits.data//3)], digits.target[:-len(digits.target//3)]
clf.fit(X,y)

print(clf.predict(digits.data[-4:-3])) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit
plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
