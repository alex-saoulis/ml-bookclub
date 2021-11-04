from pathlib import Path
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import numpy as np
from numpy import ndarray
from tensorflow.python.keras import activations
from tensorflow.python.ops.gen_array_ops import squeeze


class ManifoldExp():

    def __init__(self,classes=2, man_exp=False,train_path=None,epochs=5,model_path=None,vgg=False,img_width=224,img_height=224):

        self.img_width = img_width
        self.img_height = img_height
        self.k = None
        if man_exp:
            self.manifold_exp_list = np.load("Explorer_List.npy")
            self.class_list = np.load("Class_List.npy")
        else:
            self.manifold_exp_list = []
            self.class_list = []
            
        if not model_path == None:
            self.model = tf.keras.models.load_model(model_path)
        elif vgg:
            self.model = tf.keras.applications.vgg16.VGG16(
                weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, classes=1000,
                classifier_activation='softmax'
            )
            self.img_width = 224
            self.img_height = 224
        else: 
            self.model = Sequential([
                    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
                    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(classes)
                    ])
            self.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.epochs = epochs
        self.classes = classes
        self.train_path = train_path
        self.labels = []
        self.data = []
        self.img_height = img_height
        self.img_width = img_width
        
        
    
    def close_neighboor_avg(self,training_set:ndarray, testing_set:ndarray, training_labels:ndarray,k=7,metric:str='euclidean')->ndarray: 
        from sklearn.metrics import pairwise_distances as pdist
        import scipy.stats
        if self.k != None:
            k = self.k
        # Calculate an m x n distance matrix
        pairse_distance = pdist(testing_set,training_set,metric)

        #sort along the axis=1 so basically it sorts per row not column and than take the k closest neighboor per row to compare
        #sort_indeces = np.argsort(pairse_distance,axis=1)[:,:k]
        sort_indeces = np.argsort(pairse_distance,axis=1)[:,:]
        nearest_labels = np.squeeze(training_labels[sort_indeces])
        sorted_set = np.squeeze(training_set[sort_indeces])
        sorted_set_label = [(set_near,label) for set_near, label in zip(sorted_set,nearest_labels)]

        tmp_ref_label = nearest_labels[0]
        same_class = []
        diff_class = []
        stupid_index_same_cls = 0
        stupid_index_diff_cls = 0

        for point,label in zip(sorted_set,nearest_labels):
            if stupid_index_same_cls >= k and stupid_index_diff_cls >= k:
                break
            if tmp_ref_label == label and stupid_index_same_cls < k:
                same_class.append(point)
                stupid_index_same_cls = stupid_index_same_cls + 1
            if tmp_ref_label != label and stupid_index_diff_cls < k:
                diff_class.append(point)
                stupid_index_diff_cls = stupid_index_diff_cls + 1
        
        point_manifold_same = sum(same_class) / len(same_class)
        point_manifold_diff = sum(diff_class) / len(diff_class)

        #I am using scipy.stats.mode because my computer will break otherwise and I don't want that to happen
        return np.array(point_manifold_same), np.array(point_manifold_diff)

    def parser(self):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        self.train_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(self.img_height, self.img_width),
        batch_size=3)
        return train_ds

    def train(self):
        self.model.fit(
        self.parser(),
        epochs=self.epochs
        )
        tf.keras.models.save_model(
            self.model, "manifold", overwrite=True, include_optimizer=True, save_format=None,
            signatures=None, options=None, save_traces=True
        )
    
    def manifold_importer(self,X_path):
        import os
        import sys
        classs = 0
        for root, dirs, files in os.walk(X_path):
            toolbar_width = len(files)
            element_index = 0
            for file in files:
                sys.stdout.write('\r')
                path_point = root + "\\" + file

                img = image.load_img(path_point,target_size=(self.img_height, self.img_width, 3))

                x =tf.keras.preprocessing.image.img_to_array(
                    img, data_format=None, dtype=None
                )
                manifold_embed = self.conv_activation(x)[0]
                self.manifold_exp_list.append(manifold_embed)
                self.class_list.append(classs)
                sys.stdout.write("[%-60s] %d%%" % ("="*int(60*(element_index)/toolbar_width), (100*(element_index)/toolbar_width)))
                sys.stdout.flush()
                sys.stdout.write(", Files Done: %d"% (element_index))
                sys.stdout.flush()

            if len(files) != 0:
                classs = classs + 1

            sys.stdout.write("\n")
            sys.stdout.flush()

        self.manifold_exp_list = np.array(self.manifold_exp_list)
        self.class_list = np.array(self.class_list)
        np.save("Explorer_List",self.manifold_exp_list)
        np.save("Class_List",self.class_list)


    def manifold_importer_non_save(self,test_point,X_path,k=20,metric:str='euclidean'):
        import os 
        from sklearn.metrics import pairwise_distances as pdist
        import sys
        classs = 0
        self.k = k
        disatnces_list = []
        embed_list = []
        class_list = []
        test_embed = self.conv_activation(test_point)[0].reshape(1, -1)
        
        for root, dirs, files in os.walk(X_path):
            toolbar_width = len(files)
            element_index = 0
            for file in files:
                sys.stdout.write('\r')
                element_index = element_index + 1
                path_point = root + "\\" + file

                img = image.load_img(path_point,target_size=(self.img_height, self.img_width, 3))

                x =tf.keras.preprocessing.image.img_to_array(
                    img, data_format=None, dtype=None
                )
                manifold_embed = self.conv_activation(x)[0]
                
                if len(disatnces_list) <= self.k:
                    disatnces_list.append(pdist(test_embed.reshape(1, -1),manifold_embed.reshape(1, -1),metric))
                    embed_list.append(manifold_embed)
                    class_list.append(classs)
                else:
                    for index in range(len(disatnces_list)):
                        if disatnces_list[index] > pdist(test_embed.reshape(1, -1),manifold_embed.reshape(1, -1),metric):
                            disatnces_list[index] = pdist(test_embed.reshape(1, -1),manifold_embed.reshape(1, -1),metric)
                            embed_list[index] = manifold_embed
                            class_list[index] = classs
                            break
                sys.stdout.write("[%-60s] %d%%" % ("="*int(60*(element_index)/toolbar_width), (100*(element_index)/toolbar_width)))
                sys.stdout.flush()
                sys.stdout.write(", Files Done: %d"% (element_index))
                sys.stdout.flush()


            for embed, cla in zip(embed_list,class_list):
                self.manifold_exp_list.append(embed)
                self.class_list.append(cla)
                
            disatnces_list = []
            embed_list = []
            class_list = []
            
            if len(files) != 0:
                classs = classs + 1
                
            sys.stdout.write("\n")
            sys.stdout.flush()

        self.manifold_exp_list = np.array(self.manifold_exp_list)
        self.class_list = np.array(self.class_list)
        np.save("Explorer_List",self.manifold_exp_list)
        np.save("Class_List",self.class_list)


    def manifold_explorer(self,X,iter=200,learning_rate=0.1):
        import sys
        X_embeded = self.conv_activation(X)[0].reshape(1, -1)
        y_same,y_diff=self.close_neighboor_avg(self.manifold_exp_list,X_embeded,self.class_list,k=100)
        X_embeded = np.squeeze(X_embeded)
        vector = y_diff - y_same
        alpha = sum(np.power(vector,2))/X_embeded.shape[0]
        #alpha[alpha == 0.] = 1e300 # dumb trick to avoid zero values
        alpha = alpha/100
        y = tf.Variable(alpha*vector + X_embeded,trainable=False,dtype=tf.float32)
        img_X = tf.Variable(X,trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        number_iter=iter
        print("Building image:\n")
        for i in range(number_iter):
            sys.stdout.write('\r')
            with tf.GradientTape() as tape:
               # tape.watch(img_X)
                y_pred = self.conv_activation(img_X)[1]
                error = y - y_pred
                #loss = tf.keras.losses.MeanSquaredError(error)#+ 0.001 * tf.image.total_variation(img_X)
                loss = tf.nn.l2_loss(error) + 0.001 * tf.image.total_variation(img_X)
                #loss = tf.reduce_mean(error**2) + 0.001 * tf.image.total_variation(img_X)#
            dloss_dparams = tape.gradient(loss, [img_X])
            #img_X = img_X - learning_rate * dloss_dparams
            optimizer.apply_gradients(zip(dloss_dparams, [img_X]))
            sys.stdout.write("[%-60s] %d%%" % ("="*int(60*(i)/number_iter), (100*(i)/number_iter)))
            sys.stdout.flush()
            sys.stdout.write(", Iteration step: %d/%d  Loss: %f"% ((i),number_iter,(loss)))
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        return img_X
           

    def embed_activation(self,point):
        if len(point.shape) == 3:
            point = tf.expand_dims(point, 0)
        
        extractor = tf.keras.Model(inputs=self.model.inputs,
                        outputs=[layer.output for layer in self.model.layers])
        #Remove extra dims
        activations = [np.squeeze(embed_layer.numpy()) for embed_layer in extractor(point)]
        tf_activations = [embed_layer for embed_layer in extractor(point)]
        #Stack all elements
        tmp = activations[0].flatten()
        
        for index in range(1,len(activations)):
            tmp = np.append(tmp,activations[index])
        activations = tmp

        tmp = tf.reshape(tf_activations[0], [-1])
        for index in range(1,len(tf_activations)):
            tmp = tf.concat((tmp,tf.reshape(tf_activations[index], [-1])),axis=0)
        tf_activations = tmp

        return np.array(activations), tf_activations

    def conv_activation(self,point):
            if len(point.shape) == 3:
                point = tf.expand_dims(point, 0)
            
            out = []
            for layer in self.model.layers:
                if "conv" in layer.name:
                    out.append(layer.output)
            out = out[-4:]
            extractor = tf.keras.Model(inputs=self.model.inputs,
                            outputs=out)
            #Remove extra dims
            activations = [np.squeeze(embed_layer.numpy()) for embed_layer in extractor(point)]
            tf_activations = [embed_layer for embed_layer in extractor(point)]
            #Stack all elements
            tmp = activations[0].flatten()
            
            for index in range(1,len(activations)):
                tmp = np.append(tmp,activations[index])
            activations = tmp

            tmp = tf.reshape(tf_activations[0], [-1])
            for index in range(1,len(tf_activations)):
                tmp = tf.concat((tmp,tf.reshape(tf_activations[index], [-1])),axis=0)
            tf_activations = tmp

            return np.array(activations), tf_activations



mn = ManifoldExp(vgg=True,man_exp=True)
img = image.load_img("test_point2.jpg",target_size=(224, 224, 3))

x =tf.keras.preprocessing.image.img_to_array(
    img, data_format=None, dtype=None
)

#mn.manifold_importer_non_save(x,"data_set",k=100)
#mn.manifold_explorer(x)
#print(mn.manifold_explorer(x))

import matplotlib.pyplot as plt
#  
plt.imshow(tf.cast(x, tf.int32))
plt.show()
plt.imshow(tf.cast(mn.manifold_explorer(x), tf.int32))
plt.show()
