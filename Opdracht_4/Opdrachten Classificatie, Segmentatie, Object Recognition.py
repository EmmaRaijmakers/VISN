import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.utils import shuffle

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

#randomize data and target in the same way, to still ensure the match on the indexes
ran_data, ran_target = shuffle(digits.data, digits.target, random_state=0)

clf = svm.SVC(gamma=0.001, C=100)

#create X and y to be 2/3 of the dataset
X,y = ran_data[:-(len(ran_data)//3)], ran_target[:-(len(ran_target)//3)]

clf.fit(X,y)

#create variables to calculate accuracy
size_test_set = len(ran_data)//3
total_correct = 0

#for the items not used in the training set
for i in range((len(ran_data)//3) * 2, len(ran_data)):
    
    #if the prediction was correct, add one to total correct
    if(clf.predict(ran_data[i: i + 1])) == ran_target[i]:
        total_correct+=1

#calculate and print accuracy
print(f'accuracy == {total_correct/size_test_set}%')

# print(clf.predict(digits.data[-4:-3])) #Beetje flauw, maar hij wil perse 2d-array hebben ook als er maar 1 element inzit
# plt.imshow(digits.images[-4], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
