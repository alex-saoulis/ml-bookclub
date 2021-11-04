import numpy as np
import mnist_reader
from numpy import ndarray
#from sklearn.neighbors import NeighborhoodComponentsAnalysis

X_train, y_train = mnist_reader.load_mnist('', kind='train')
X_test, y_test = mnist_reader.load_mnist('', kind='t10k')

def cloth_matrix(value,title) -> None:
    import matplotlib.pyplot as plt

    cloth_dict = {
        0: 'T-shirt/top',
        1:'Trouser',
        2:'Pullover',
        3:'Dress',
        4:'Coat',
        5:'Sandal',
        6:'Shirt',
        7:'Sneaker',
        8:'Bag',
        9:'Ankle boot',
        10: 'Weight Vector'
    }

    count = 1
    num_matrix = []
    tmp_num_mat = []
    for element in value:
        if count % 28 == 0:
            num_matrix.append(tmp_num_mat)
            tmp_num_mat = []
        else:
            tmp_num_mat.append(element)
        count = count + 1
    title_str = 'The clothing shown: ' + cloth_dict[title]
 
    plt.matshow(num_matrix) 
    plt.gray()
    plt.title(title_str)
    plt.show()

class Perceptron(object):
    
    def __init__(self,y_1:int=None,y_2:int=None):
        import os.path as p
        name_save = "weights_"+ str(y_1) + "_" + str(y_2) + ".npy"
        path = 'weights/'+name_save
        if p.isfile(path):
            self.W = np.load(path)
        else:
            self.W = 0
        if p.isfile('weights_history.npy'):
            self.W_history = np.load('weights_history.npy')
        


    def train(self,training_set:ndarray,training_labels:ndarray,y_1:int,y_2:int) ->None:
        import time
        start_time = time.time()
        self.t_flag = False
        self.W = np.zeros(training_set.shape[1])
        index = 0
        self.y_1 = y_1
        self.y_2 = y_2
        m = 0
        while True:

            m = 0

            for index in range(training_set.shape[0]):
                
                if training_labels[index] == y_1:
                    y_tmp = 1
                elif training_labels[index] == y_2:
                    y_tmp = -1
                else:
                    continue
           
                if np.dot(self.W,training_set[index])*y_tmp <= 0:
                    self.W = self.W + training_set[index]*y_tmp
                    m = m + 1
            if m == 0 or time.time() - start_time > 600:
                if m != 0:
                    self.t_flag = False
                else:
                    self.t_flag = True
                    name_save = "weights_"+ str(y_1) + "_" + str(y_2) + ".npy"
                    np.save('weights/'+name_save,self.W)
                break
    
    def train_min_best(self,training_set:ndarray,training_labels:ndarray,y_1:int,y_2:int) ->None:
        import time
        import os.path as p

        start_time = time.time()
        self.t_flag = False
        name_save = "weights_"+ str(y_1) + "_" + str(y_2) + ".npy"
        path = 'weights/'+name_save
        if p.isfile(path):
            return
        else:
            self.W = np.zeros(training_set.shape[1])
        
        index = 0
        self.y_1 = y_1
        self.y_2 = y_2
        m = 0
        minM = 1000000
        while True:
            
            m = 0

            for index in range(training_set.shape[0]):
                
                if training_labels[index] == y_1:
                    y_tmp = 1
                elif training_labels[index] == y_2:
                    y_tmp = -1
                else:
                    continue
           
                if np.dot(self.W,training_set[index])*y_tmp <= 0:
                    self.W = self.W + training_set[index]*y_tmp
                    m = m + 1
            if m < minM:
                wMin = self.W
                minM = m

            if m == 0 or time.time() - start_time > 300:
                if m != 0:
                    self.t_flag = True
                    name_save = "weights_"+ str(y_1) + "_" + str(y_2) + ".npy"
                    np.save('weights/'+name_save, wMin)
                else:
                   
                    self.t_flag = True
                    name_save = "weights_"+ str(y_1) + "_" + str(y_2) + ".npy"
                    np.save('weights/'+name_save, self.W)
                break

    def train_one_vs_all(self,training_set:ndarray,training_labels:ndarray,y_1:int) ->None:
        import time
        start_time = time.time()
        self.t_flag = False
        self.W = np.zeros(training_set.shape[1])
        index = 0
        self.y_1 = y_1
        m = 0
     
        while True:

            m = 0

            for index in range(training_set.shape[0]):
                
                if training_labels[index] == y_1:
                    y_tmp = 1
                else:
                    y_tmp = -1
           
                if np.dot(self.W,training_set[index])*y_tmp <= 0:
                    self.W = self.W + training_set[index]*y_tmp
                    m = m + 1
            if m == 0 or time.time() - start_time > 600:
                if m != 0:
                    self.t_flag = False
                else:
                    self.t_flag = True
                    name_save = "weights_"+ str(y_1) + "_" + "all"+ ".npy"
                    np.save('weights/'+name_save,self.W)
                break
        
    def train_exp(self,training_set:ndarray,training_labels:ndarray,y_1:int,y_2:int) ->None:
        
        self.W = np.zeros(training_set.shape[1])
        index = 0
        self.y_1 = y_1
        self.y_2 = y_2
        m = 0
        while True:

            m = 0

            for index in range(training_set.shape[0]):
                
                if training_labels[index] == y_1:
                    y_tmp = 1
                elif training_labels[index] == y_2:
                    y_tmp = -1
                else:
                    continue
        
                if np.dot(self.W,training_set[index])*y_tmp <= 0:
                    self.W = self.W + training_set[index]*y_tmp
                    m = m + 1
            if m == 0 :
                break

    def train_history(self,training_set:ndarray,training_labels:ndarray,y_1:int,y_2:int) ->None:
        self.W = np.zeros(training_set.shape[1])
        index = 0
        W_history = []
        
        while True:
            m = 0
         
            for index in range(training_set.shape[0]):
                if training_labels[index] == y_1:
                    y_tmp = 1
                elif training_labels[index] == y_2:
                    y_tmp = -1
                else:
                    continue
                if np.dot(self.W,training_set[index])*y_tmp <= 0:
                    W_history.append(self.W)
                    self.W = self.W + training_set[index]*y_tmp
                    m = m + 1
            if m == 0:
                break
        np.save('weights_history.npy',W_history)
        np.save('weights.npy',self.W)
        self.W_history = W_history
        return self.W_history

    def test(self,testing_set:ndarray) ->None:
        index = 0
        self.output=[]
        for index in range(testing_set.shape[0]):
            if np.dot(self.W,testing_set[index]) < 0:
                self.output.append(-1)
            elif np.dot(self.W,testing_set[index]) > 0:
                self.output.append(1)
        self.output = np.array(self.output)
        return self.output

    def test_one_vs_one(self,testing_set:ndarray) ->None:
        import os.path as p
        from scipy import stats
        index = 0
        final = []
        y1list = []
        y2list = []
        for index in range(testing_set.shape[0]):
            self.output = []
            for y_1 in range(6):
                for y_2 in range(10):
                   
                    if y_1 == y_2:
                        continue
                    y1list.append(y_1)
                    y2list.append(y_2)
                    name_save = "weights_"+ str(y_1) + "_" + str(y_2) + ".npy"
                    path = 'weights/'+name_save
                    self.W = np.load(path)
                    if np.dot(self.W,testing_set[index]) < 0:
                        self.output.append(y_2)
                    elif np.dot(self.W,testing_set[index]) > 0:
                        self.output.append(y_1)

            final.append(np.squeeze(stats.mode(np.array(self.output),axis=None))[0])
        return np.array(final)
        




def sample_indices(labels, *num_per_class) -> ndarray:

    indices = []
    lab = np.copy(labels)
    for index in range(len(lab)):
        if lab[index] == -1:
            lab[index] = 0
    for cls, num in enumerate(num_per_class):
        cls_indices = np.where(lab == cls)[0]
        indices.extend(np.random.choice(cls_indices, size=num, replace=False))
    return np.array(indices)

def pre_process_test(data_set:ndarray,label_set:ndarray,y_1:int,y_2:int):
    out = []
    out_set = []
    for index in range(data_set.shape[0]):
        if label_set[index] == y_1:
            out.append(1)
            out_set.append(data_set[index])
        elif label_set[index] == y_2:
            out.append(-1)
            out_set.append(data_set[index])
    out = np.array(out)
    out_set = np.array(out_set)
    return out,out_set


def generator():
    for i in range(10):
        for j in range(10):
            if i == j:
                continue
            else:
                p = Perceptron(i,j)
                p.train_min_best(X_train,y_train,i,j)
                if p.t_flag:
                    test_label,test_set = pre_process_test(X_test,y_test,i,j)
                    sum(test_label==p.test(test_set))/len(p.test(test_set))
                    print('Accuracy: %s and %s',i,j,sum(test_label==p.test(test_set))/len(p.test(test_set)))
                    with open('results.txt', 'a') as the_file:
                        res = 'Accuracy for: ' + str(i) + ' '+  str(j) +' '+ str(sum(test_label==p.test(test_set))/len(p.test(test_set)))
                        the_file.write(res+'\n')
                else:
                    print('skipped')


def generator_one_vs_all(p:Perceptron):
    for i in range(10):
        p.train_one_vs_all(X_train,y_train,i)
        if p.t_flag:
            test_label,test_set = pre_process_test(X_test,y_test,i,j)
            sum(test_label==p.test(test_set))/len(p.test(test_set))
            print('Accuracy: ',sum(test_label==p.test(test_set))/len(p.test(test_set)))
            with open('results.txt', 'a') as the_file:
                res = 'Accuracy for: ' + str(i) + ' '+  "all"+' '+ str(sum(test_label==p.test(test_set))/len(p.test(test_set)))
                the_file.write(res+'\n')
        else:
            print('skipped')


def exp1():
    number = []
    acurracy = []
    p = Perceptron()
    train_label,train_data = pre_process_test(X_train,y_train,2,5)
    import matplotlib.pyplot as plt
    for i in range(20):
        number.append(i)
        trainingIndex = sample_indices(train_label, 80, 80)
        testingIndex = np.delete(np.arange(len(train_label)), trainingIndex)
        tset = train_data[trainingIndex]
        tlabel = train_label[trainingIndex]
        p.train_exp(tset,tlabel,1,-1)
        test = train_data[testingIndex]
        p_output = p.test(test)   
        label = train_label[testingIndex]
        o=sum(label==p_output)/len(p_output)
        acurracy.append(o)
    print("List of acurracy: ",acurracy)
    print("Mean: ",np.mean(acurracy))
    print("Standard deviation: ",np.std(acurracy))

    plt.plot(np.array(number),np.array(acurracy))
    plt.show()

# The multiclass output
p = Perceptron()
pout = p.test_one_vs_one(X_test)
print('Shape pout',pout.shape)
print('Shape Y test',y_test.shape)
print('pout ',pout)
print('Y test',y_test)
so = sum(y_test==pout)/len(pout)
print(so)

# The binary classification with 0 and 1
p = Perceptron(0,1)
test_label,test_set = pre_process_test(X_test,y_test,0,1)
indecis = (test_label==p.test(test_set)).nonzero()

cloth_matrix(p.W,10)

sum(test_label==p.test(test_set))/len(p.test(test_set))
print('Accuracy Final: ',sum(test_label==p.test(test_set))/len(p.test(test_set)))