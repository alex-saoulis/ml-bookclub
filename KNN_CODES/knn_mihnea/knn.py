from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import pairwise_distances as pdist
from scipy.sparse import csr_matrix

def parser(file:str) -> ndarray : 
    file1 = open(file, 'r') 
    Lines = file1.readlines()  
    labels = []
    drawings = []
    for line in Lines: 
        draw = list(map(int,line.split(',')))
        if len(draw) == 65:
            tmp_label = draw[-1]
            del draw[-1]
            labels.append(tmp_label)
        drawings.append(draw)
    return np.array(drawings), np.array(labels)

def sample_indices(labels, *num_per_class) -> ndarray:

    indices = []
    for cls, num in enumerate(num_per_class):
        cls_indices = np.where(labels == cls)[0]
        indices.extend(np.random.choice(cls_indices, size=num, replace=False))
    return np.array(indices)

def number_matrix(value,title) -> None:
    count = 1
    num_matrix = []
    tmp_num_mat = []
    for element in value:
        if count % 8 == 0:
            num_matrix.append(tmp_num_mat)
            tmp_num_mat = []
        else:
            tmp_num_mat.append(element)
        count = count + 1
    title_str = 'The number shown: ' + str(title)
    plt.title(title_str)
    plt.gray()
    plt.matshow(num_matrix) 
    plt.show()



def minkowski_distance(training_set:ndarray, testing_set:ndarray,p:int):
    final_sum_list = []
    for test_point in testing_set:
        final_sum_list.append(np.power(np.sum(np.power(abs(training_set - test_point),p),axis=1),1/p))
    return np.array(final_sum_list)


def knn_minkowski(training_set:ndarray, testing_set:ndarray, training_labels:ndarray,k:int=1,p:int=1)->ndarray: 
    # Calculate an m x n distance matrix
    pairse_distance = minkowski_distance(training_set,testing_set,p)

    #sort along the axis=1 so basically it sorts per row not column and than take the k closest neighboor per row to compare
    sort_indeces = np.argsort(pairse_distance,axis=1)[:,:k]
    nearest_labels = training_labels[sort_indeces]

    #I am using scipy.stats.mode because my computer will break otherwise and I don't want that to happen
    return np.squeeze(scipy.stats.mode(nearest_labels,axis=1))[0]

def knn_general(training_set:ndarray, testing_set:ndarray, training_labels:ndarray,k:int=1,metric:str='euclidean')->ndarray: 
    # Calculate an m x n distance matrix
    pairse_distance = pdist(testing_set,training_set,metric)

    #sort along the axis=1 so basically it sorts per row not column and than take the k closest neighboor per row to compare
    sort_indeces = np.argsort(pairse_distance,axis=1)[:,:k]
    nearest_labels = training_labels[sort_indeces]

    #I am using scipy.stats.mode because my computer will break otherwise and I don't want that to happen
    return np.squeeze(scipy.stats.mode(nearest_labels,axis=1))[0]



train_data = parser('optdigits.tra')[0]
train_label = parser('optdigits.tra')[1]
test_data = parser('optdigits.tes')[0]
test_label = parser('optdigits.tes')[1]
acurracy = []
for i in range(20):
    trainingIndex = sample_indices(train_label, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300)
    testingIndex = np.delete(np.arange(len(train_label)), trainingIndex)
    knn_output = knn_general(train_data[trainingIndex],train_data[testingIndex],train_label[trainingIndex])   
    label = train_label[testingIndex]
    o=sum(train_label[testingIndex]==knn_output)/len(knn_output)
    acurracy.append(o)
print("List of acurracy: ",acurracy)
print("Mean: ",np.mean(acurracy))
print("Standard deviation: ",np.std(acurracy))

train_data = parser('optdigits.tra')[0]
train_label = parser('optdigits.tra')[1]
acurracy = []
for i in range(20):
    trainingIndex = sample_indices(train_label, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300)
    testingIndex = np.delete(np.arange(len(train_label)), trainingIndex)
    knn_output = knn_minkowski(train_data[trainingIndex],train_data[testingIndex],train_label[trainingIndex],p=1,k=1)   
    label = train_label[testingIndex]
    o=sum(train_label[testingIndex]==knn_output)/len(knn_output)
    acurracy.append(o)
print("List of acurracy: ",acurracy)
print("Mean: ",np.mean(acurracy))
print("Standard deviation: ",np.std(acurracy))

knn_output = knn_minkowski(train_data,test_data,train_label,p=2)
o = sum(test_label==knn_output)/len(knn_output)
print('Minkiwski acc: ',o)

knn_output = knn_minkowski(train_data,test_data,train_label)
o = sum(test_label==knn_output)/len(knn_output)
print('General acc: ',o)

number_matrix(train_data[56],train_label[56])