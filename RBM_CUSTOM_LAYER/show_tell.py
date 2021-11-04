#from ptsne import  GaussianBernouli, BernouliBoltzmanMachine, GaussianBoltzmanMachine, TSNE_Layer
import numpy as np
import sys
import tensorflow as tf

class Simple(tf.keras.Model):

  def __init__(self):
    super(Simple, self).__init__()
    self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(10)
    self.dense3 = tf.keras.layers.Dense(40)
    self.dense4 = tf.keras.layers.Dense(200)
    self.dense5 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)


class GaussianBoltzmanMachine(tf.keras.layers.Layer):

    def __init__(self,
                num_visible,
                num_hidden,
                num_iterations=100,
                type_dist=["bernouli","gaussian"],
                batch_size=100,
                weight_cost=0.0002,
                verbose=1):
        super(GaussianBoltzmanMachine, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_iterations = num_iterations
        self.distribution_a = type_dist[0]
        self.distribution_b = type_dist[1]
        #self.W = tf.Variable(tf.random.uniform([num_visible, num_hidden]) * 0.1 , name="weights",dtype=tf.float32)
        self.W = tf.Variable(tf.random.normal([self.num_visible, self.num_hidden], mean=0., stddev=1),
            trainable=True, name="weights")
        self.a = tf.Variable(tf.zeros([1, num_visible], name = "visible_biases"),
            trainable=True,dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1, num_hidden], name = "hidden_biases"),
            trainable=True,dtype=tf.float32)
        self.deltaW = tf.Variable(tf.zeros([num_visible, num_hidden]),
            trainable=True,name="delta_weights")
        self.delta_a = tf.Variable(tf.zeros([1, num_visible]),
            trainable=True,name="delta_bias_vizible")
        self.delta_b = tf.Variable(tf.zeros([1, num_hidden]),
            trainable=True,name="delta_bias_hidden")

        self.batch_size = batch_size
        self.verbose=verbose

        self.weight_cost = weight_cost
        self.current_iter = 0
        self.initial_moment = 0.5
        self.final_moment = 0.9
        self.current_moment = self.initial_moment
        self.eta = 0.001 # I am dumb, it actuallymakes sense, see previous commit to clarify

    def get_prob_gaussian(self, layer):
        if layer == "visible":
            prob_viz = tf.nn.sigmoid(tf.matmul(self.h, tf.transpose(self.W,conjugate=True)) + self.a)
            #return tf.clip_by_value(prob_viz, clip_value_min=1e-10,clip_value_max=1000)
            return prob_viz
        elif layer == 'hidden':
            prob_hidden = tf.matmul(self.v,self.W) + self.b
            return prob_hidden
            #return tf.clip_by_value(prob_hidden,clip_value_min=1e-10,clip_value_max=1000)

    def sample_gaussian(self,probabilities, stddev=1.0):
        return tf.add(probabilities, tf.random.normal(tf.shape(probabilities), mean=0.0, stddev=stddev))

    # def build(self, input_shape):
    #     print(input_shape)
    #     super(Reconstruction_Layer, self).build(input_shape)

    def train(self,data_set):
        "Jesus here it's where the fun starts"
        for itera in range(self.num_iterations):
           
            print("Iteration: % d out of % d" %(itera+1,self.num_iterations))
            indices = np.random.permutation(data_set.shape[0])

            if itera <= 5:
                self.current_moment = self.initial_moment
            else:
                self.current_moment = self.final_moment
            for batch in range(1,data_set.shape[0],self.batch_size):
                sys.stdout.write('\r')
                #This is the Gibbs sampeling mentioned in the paper. I wont use the function made initially, because honestly he does it kind of strange.
                
                #Initial vis layer
                self.v = tf.Variable(tf.convert_to_tensor(data_set[indices[batch:min(batch+self.batch_size - 1,data_set.shape[0])]][:],dtype=tf.float32),name='batch')
              
                initial_v = self.v
                
                #Sample probs for the hidden layer
                prob_h = self.get_prob_gaussian("hidden")

                #Sample states for the hidden layer
                hid_states = self.sample_gaussian(prob_h)
                self.h = hid_states

                #Compute probabilities for visible nodes
                v2 = self.get_prob_gaussian("visible")
                self.v = v2
               # print(v2)
                #Compute probabilities for hidden nodes
                prob_hidden_2 = self.get_prob_gaussian("hidden")

                #Weight updates ???
                posprods = tf.matmul(tf.transpose(initial_v,conjugate=True),prob_h)
                negprods = tf.matmul(tf.transpose(v2,conjugate=True),prob_hidden_2)
                
                self.deltaW = self.current_moment * self.deltaW + self.eta * (((posprods - negprods) / self.batch_size) - (self.weight_cost * self.W))
                self.delta_b = self.current_moment * self.delta_b + (self.eta / self.batch_size) *  (tf.reduce_sum(prob_h,axis=0) - tf.reduce_sum(prob_hidden_2, axis=0))
                self.delta_a = self.current_moment * self.delta_a + (self.eta / self.batch_size) * (tf.reduce_sum(initial_v, axis=0) - tf.reduce_sum(v2, axis=0))

                self.W = self.W + self.deltaW
                self.a = self.a + self.delta_a
                self.b = self.b + self.delta_b
                #print(data_set.shape[0])
                sys.stdout.write("[%-60s] %d%%" % ("="*int(60*(batch)/data_set.shape[0]), (100*(batch)/data_set.shape[0])))
                sys.stdout.flush()
                sys.stdout.write(", Batch %d"% (batch/self.batch_size+1))
                sys.stdout.flush()
           
            sys.stdout.write("\n")
            sys.stdout.flush()
        #print("\n")
        return self
    def call(self,inputs):
            prob_hidden = tf.matmul(inputs,self.W) + self.b
            return prob_hidden
            
    def get_feature_map(self):
        ''' Map probs of hidden features'''
        ft_map = []
        for k in range(self.num_visible):
            self.v = tf.expand_dims(tf.one_hot(k+1, self.num_visible),0)
            ft_map.append(self.get_prob_gaussian('hidden'))
        return ft_map

class BernouliBoltzmanMachine(tf.keras.layers.Layer):

    def __init__(self,
                num_visible,
                num_hidden,
                num_iterations=100,
                type_dist=["bernouli","gaussian"],
                batch_size=100,
                weight_cost=0.0002,
                verbose=1):
        super(BernouliBoltzmanMachine, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_iterations = num_iterations
        self.distribution_a = type_dist[0]
        self.distribution_b = type_dist[1]
        #self.W = tf.Variable(tf.random.uniform([num_visible, num_hidden]) * 0.1 , name="weights",dtype=tf.float32)
        self.W = tf.Variable(tf.random.normal([self.num_visible, self.num_hidden], mean=0., stddev=1),
            trainable=True, name="weights")
        self.a = tf.Variable(tf.zeros([1, num_visible], name = "visible_biases"),
            trainable=True,dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1, num_hidden], name = "hidden_biases"),
            trainable=True,dtype=tf.float32)
        self.deltaW = tf.Variable(tf.zeros([num_visible, num_hidden]),
            trainable=True,name="delta_weights")
        self.delta_a = tf.Variable(tf.zeros([1, num_visible]),
            trainable=True,name="delta_bias_vizible")
        self.delta_b = tf.Variable(tf.zeros([1, num_hidden]),
            trainable=True,name="delta_bias_hidden")

        self.batch_size = batch_size
        self.verbose=verbose

        self.weight_cost = weight_cost
        self.current_iter = 0
        self.initial_moment = 0.5
        self.final_moment = 0.9
        self.current_moment = self.initial_moment
        self.eta = 0.001 # I am dumb, it actuallymakes sense, see previous commit to clarify

    def get_prob_bernouli(self, layer):
        if layer == "visible":
            prob_viz = tf.nn.sigmoid(tf.matmul(self.h, tf.transpose(self.W)) + self.a)
            return prob_viz
        elif layer == 'hidden':
            prob_hidden = tf.nn.sigmoid(tf.matmul(self.v,self.W) + self.b)
            return prob_hidden

    def sample_bernouli(self,probabilities):
        return tf.floor(probabilities + tf.random.uniform(tf.shape(probabilities), 0, 1))

    def train(self,data_set):
        "Jesus here it's where the fun starts"
        for itera in range(self.num_iterations):
            
            print("Iteration: % d out of % d" %(itera+1,self.num_iterations))
            indices = np.random.permutation(data_set.shape[0])
           
            if itera <= 5:
                self.current_moment = self.initial_moment
            else:
                self.current_moment = self.final_moment
            for batch in range(1,data_set.shape[0],self.batch_size):
                sys.stdout.write('\r')
                #This is the Gibbs sampeling mentioned in the paper. I wont use the function made initially, because honestly he does it kind of strange.
                #Initial vis layer
                self.v = tf.Variable(tf.convert_to_tensor(data_set[indices[batch:min(batch+self.batch_size - 1,data_set.shape[0])]][:],dtype=tf.float32),name='batch')
                initial_v = self.v

                #Sample probs for the hidden layer
                prob_h = self.get_prob_bernouli("hidden")

                #Sample states for the hidden layer
                hid_states = self.sample_bernouli(prob_h)
                self.h = hid_states

                #Compute probabilities for visible nodes
                v2 = self.get_prob_bernouli("visible")
                self.v = v2

                #Compute probabilities for hidden nodes
                prob_hidden_2 = self.get_prob_bernouli("hidden")

                #Weight updates ???
                posprods = tf.matmul(tf.transpose(initial_v,conjugate=True),prob_h)
                negprods = tf.matmul(tf.transpose(v2,conjugate=True),prob_hidden_2)
        
                self.deltaW = self.current_moment * self.deltaW + self.eta * (((posprods - negprods) / self.batch_size) - (self.weight_cost * self.W))
                self.delta_b = self.current_moment * self.delta_b + (self.eta / self.batch_size) * (tf.reduce_sum(prob_h,axis=0) - tf.reduce_sum(prob_hidden_2, axis=0))
                self.delta_a = self.current_moment * self.delta_a + (self.eta / self.batch_size) * (tf.reduce_sum(initial_v, axis=0) - tf.reduce_sum(v2, axis=0))
                
                self.W = self.W + self.deltaW
                self.a = self.a + self.delta_a
                self.b = self.b + self.delta_b


                sys.stdout.write("[%-60s] %d%%" % ("="*int(60*(batch)/data_set.shape[0]), (100*(batch)/data_set.shape[0])))
                sys.stdout.flush()
                sys.stdout.write(", Batch %d"% (batch/self.batch_size+1))
                sys.stdout.flush()
           
            sys.stdout.write("\n")
            sys.stdout.flush()
        # print("\n")       
        return self
    def call(self,inputs):
            prob_hidden = tf.nn.sigmoid(tf.matmul(inputs,self.W) + self.b)
            return prob_hidden

    def get_feature_map(self):
        ''' Map probs of hidden features'''
        ft_map = []
        for k in range(self.num_visible):
            self.v = tf.expand_dims(tf.one_hot(k+1, self.num_visible),0)
            ft_map.append(self.get_prob_bernouli('hidden'))
        return ft_map

class GaussianBernouli(GaussianBoltzmanMachine):
    "Some improvements can be made for tha training stage such as removing the mean for all inputs. Will add functionality later"
    #TODO: 1.add std value momnetum for better training
    #      2. add sparse hidden layer for better feature extraction
    def get_prob_gaussian(self, layer):
        if layer == "visible":
            viz_mean = tf.matmul(self.h, tf.transpose(self.W)) + self.a
            prob_viz = tf.random.normal([1,self.num_visible], mean=viz_mean,stddev=0.1)
            return prob_viz
        elif layer == 'hidden':
            prob_hidden = tf.nn.sigmoid(tf.matmul(self.v,self.W) + self.b)
            return prob_hidden

    def train(self,data_set):
        # This is so dumb, but it's made by me so what do you expect, I will fix it and improve it once I know that the RBM PTSNE works. Reason: to use the tf framework for gpu calculation, numpy wouldn't have cut it
        pre_process_data = tf.convert_to_tensor(data_set,dtype=tf.float32)
        data_mean, data_var = tf.nn.moments(pre_process_data, [1], keepdims =True)
        inputs_tf = tf.math.divide(tf.math.subtract(pre_process_data, data_mean), data_var)
        inputs = inputs_tf.numpy()
        super().train(inputs)

    def call(self, inputs):
        pre_process_data = inputs
        data_mean, data_var = tf.nn.moments(pre_process_data, [1], keepdims =True)
        inputs_tf = tf.math.divide(tf.math.subtract(pre_process_data, data_mean), data_var)
        prob_hidden = tf.nn.sigmoid(tf.matmul(inputs_tf,self.W) + self.b)
        return prob_hidden
    




class ComplicatedModel(tf.keras.Model):

    def __init__(self,rbm_layer=["gaussian_bernouli","bernouli","bernouli","bernouli","gaussian"],
                 out_size=[64,40,40,200,10],
                 rbm_iter=[200,100,100,90,90]):
        super(ComplicatedModel, self).__init__()
        self.rbm_layer = rbm_layer
        self.out_size = out_size
        self.rbm_iter = rbm_iter
        self.real_layer = None
    
    def _layer_pick(self,layer_type,input_shape,hidden,iter_index):
        if layer_type == "bernouli":
            if self.rbm_iter is not None:
                return BernouliBoltzmanMachine(input_shape,hidden,num_iterations=self.rbm_iter[iter_index])
            else:
                return BernouliBoltzmanMachine(input_shape,hidden)
        elif layer_type == "gaussian":
            if self.rbm_iter is not None:
                return GaussianBoltzmanMachine(input_shape,hidden,num_iterations=self.rbm_iter[iter_index])
            else:
                 return GaussianBoltzmanMachine(input_shape,hidden)
        elif layer_type == "gaussian_bernouli":
            if self.rbm_iter is not None:
                return GaussianBernouli(input_shape,hidden,num_iterations=self.rbm_iter[iter_index])
            else:
                return GaussianBernouli(input_shape,hidden,num_iterations=50)

    def _greedy_train(self,X):
        rbm_list = []
        for index,layer_type in enumerate(self.rbm_layer):
            if index == 0:
                rbm_list.append(self._layer_pick(layer_type,X.shape[1],self.out_size[index],index))
            else:
                rbm_list.append(self._layer_pick(layer_type,self.out_size[index-1],self.out_size[index],index))
            print("Rbm W size: ", rbm_list[index].num_visible,rbm_list[index].num_hidden)
            
        rbm_list[0].train(X)
        out_X = rbm_list[0](X).numpy()
        for index in range(1,len(rbm_list)):
            print("RBM layer: ",index)
            print("Rbm W size: ", rbm_list[index].num_visible,rbm_list[index].num_hidden)
            rbm_list[index].train(out_X)
            out_X = rbm_list[index](out_X).numpy()
        self.real_layer = rbm_list
    
    def fit(self,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          shuffle=True,
          ):
        self._greedy_train(x)
        super().fit(x=x,
          y=y,
          epochs=epochs,
          shuffle=shuffle,batch_size=batch_size)
    
    def call(self,X):
        out = self.real_layer[0](X)
        for index in range(1,len(self.real_layer)):
            out = self.real_layer[index](out)
        return out

example = ComplicatedModel(rbm_layer=["gaussian_bernouli","bernouli","gaussian"],out_size=[64,100,10],rbm_iter=[10,10,10])

example.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
train_samples = 5000
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

X_train = tf.convert_to_tensor(
    X_train, dtype=tf.float32, dtype_hint=None, name=None
)
X_test = tf.convert_to_tensor(
    X_test, dtype=tf.float32, dtype_hint=None, name=None
)

y_train = tf.convert_to_tensor(
    y_train, dtype=tf.float32, dtype_hint=None, name=None
)
y_test = tf.convert_to_tensor(
    y_test, dtype=tf.float32, dtype_hint=None, name=None
)




print(X)
print(y)


example.fit(
    X_train,y_train,
    epochs=6
)

simple = Simple()

simple.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
print("======================================================================================================")
simple.fit(
    X_train,y_train,
    epochs=6
)

tf.keras.utils.plot_model(
    simple, to_file='model2.png', show_shapes=True, show_dtype=True,
    show_layer_names=True, rankdir='TB', expand_nested=True, dpi=96
)


input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
x = tf.keras.layers.Embedding(
    output_dim=512, input_dim=10000, input_length=100)(input)
x = tf.keras.layers.LSTM(32)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
model = tf.keras.Model(inputs=[input], outputs=[output])
dot_img_file = 'model_te.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)