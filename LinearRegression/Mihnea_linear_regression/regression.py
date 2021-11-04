import os
import sys
import numpy as np
import cv2 as cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy import ndarray

def show_face(value,val1,val2) -> None:
    import matplotlib.pyplot as plt
    count = 1
    num_matrix = []
    tmp_num_mat = []
    count1 = 1
    num_matrix1 = []
    tmp_num_mat1 = []
    count2 = 1
    num_matrix2 = []
    tmp_num_mat2 = []
    for element in value:
        if count % 92 == 0:
            num_matrix.append(tmp_num_mat)
            tmp_num_mat = []
        else:
            tmp_num_mat.append(element)
        count = count + 1
    for element in val1:
        if count1 % 92 == 0:
            num_matrix1.append(tmp_num_mat1)
            tmp_num_mat1 = []
        else:
            tmp_num_mat1.append(element)
        count1 = count1 + 1
    for element in val2:
        if count2 % 92 == 0:
            num_matrix2.append(tmp_num_mat2)
            tmp_num_mat2 = []
        else:
            tmp_num_mat2.append(element)
        count2 = count2 + 1
    # title_str = 'The number shown: ' + str(title)
    # plt.title(title_str)
    # plt.gray()
    # plt.matshow(num_matrix) 
    # plt.show()

    fig, axs = plt.subplots(3)
    plt.gray()
    fig.suptitle('Reconstructed faces')
    axs[0].matshow(num_matrix)
    axs[1].matshow(num_matrix1)
    axs[2].matshow(num_matrix2)
    plt.show()



def load_images_from_folder(folder_name):
    images = []
    label = []
    label_value = 1
    for folder in os.listdir(folder_name):
        for filename in os.listdir(os.path.join(folder_name,folder)):
            
            img = cv2.imread(os.path.join(folder_name,folder, filename),cv2.IMREAD_GRAYSCALE)
            if img is not None:
               
                images.append(img.flatten())
                label.append(label_value)
                #112, 92
        label_value = label_value + 1
    return np.array(images,dtype=np.float64), np.array(label)


def train_test_split(data,labels):
    indices = np.random.permutation(data.shape[0])
    training_idx, test_idx = indices[:80], indices[80:]
    training, test = data[training_idx,:], data[test_idx,:]
    training_labels,testing_labels =  labels[training_idx,:], labels[test_idx,:]
    return training, test,training_labels,testing_labels

def train_test_split_classification(data,labels):
    indices = np.random.permutation(data.shape[0])
    training_idx, test_idx = indices[:80], indices[80:]
    training, test = data[training_idx,:], data[test_idx,:]
    training_labels,testing_labels =  labels[training_idx], labels[test_idx]
    return training, test,training_labels,testing_labels

class Regression(object):
    def __init__(self):
        self.X=None
        self.y=None
        self.w=None
        
    def train(self,tr_data, tr_labels, lambda_user=0):
        """
        A summary of your function goes here.

        data: type and description of "data"
        labels: type and description of "labels"
        lambda_user: user defined parameter for regularisation default 0
    
        Returns: type and description of the returned variable(s).
        """
        X, y = tr_data, tr_labels
        self.X, self.y = tr_data,tr_labels
        
        #X_tilde = np.hstack((np.ones((X.shape[0],1),dtype=int),X))

        # Compute the coefficient vector.
        if lambda_user == 0:
            self.w = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
        else :
            self.w = np.linalg.inv((np.transpose(X)@X+lambda_user*np.eye(X.shape[1],dtype=int)))@np.transpose(X)@y
        
        # Return model parameters.
        return self.w


    def predict(self, data ):
        """
        A summary of your function goes here.

        data: type and description of "data"

        Returns: type and description of the returned variable(s).
        """
    
        X = data
        #X_tilde = np.hstack((np.ones((X.shape[0],1),dtype=int),X))
 
        predicted_y = X@self.w
        
        return predicted_y

    def knn_general(self,training_set:ndarray, testing_set:ndarray, training_labels:ndarray,k:int=1,metric:str='euclidean',lambda_user=0)->ndarray: 
        from sklearn.metrics import pairwise_distances as pdist
        from scipy.sparse import csr_matrix
        import scipy.stats
        # Calculate an m x n distance matrix
        self.train(training_set,training_labels,lambda_user)
        pairse_distance = pdist(self.predict(testing_set).reshape(-1, 1),self.y.reshape(-1, 1),metric)

        #sort along the axis=1 so basically it sorts per row not column and than take the k closest neighboor per row to compare
        sort_indeces = np.argsort(pairse_distance,axis=1)[:,:k]
        nearest_labels = self.y[sort_indeces]

        #I am using scipy.stats.mode because my computer will break otherwise and I don't want that to happen
        return np.squeeze(scipy.stats.mode(nearest_labels,axis=1))[0]

reg = Regression()
#print(load_images_from_folder("data")[0])
init_set, init_label = np.hsplit(load_images_from_folder("data")[0],2)
tset,teset,tlab,telab=train_test_split(init_set,init_label)
atset,ateset,atlab,atelab=train_test_split_classification(load_images_from_folder("data")[0],load_images_from_folder("data")[1])
print(tset.shape)
print(tlab.shape)
print(teset.shape)
print(reg.train(tset,tlab).shape)
yte = reg.predict(teset)
print(yte.shape)
result = np.hstack((teset*1200,yte))
print(teset.dtype)
print(yte.dtype)
print(result[0])
ask = reg.knn_general(atset,ateset,atlab,k=1,metric="cosine",lambda_user=0.5)
o = sum(atelab==ask)/len(ask)
print("Class of image: ",o)
show_face(result[0],teset[0],yte[0])
show_face(result[12],teset[12],yte[12])



