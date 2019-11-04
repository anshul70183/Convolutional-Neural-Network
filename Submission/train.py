# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:43:04 2019

@author: Anshul
"""

## importing libraries
import skimage as sk
import random
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops import nn_ops, gen_nn_ops
from tensorflow.python.framework import ops
from scipy.ndimage.filters import gaussian_filter

#parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type = float, default = 0.001)
parser.add_argument("--batch_size", type = int, default = 10)
parser.add_argument("--init", type = int, default = 1)
parser.add_argument("--save_dir", type = str, default = './save_dir/')
parser.add_argument("--epochs", type = int, default = 5)
parser.add_argument("--dataAugment", type = int, default = 0)
parser.add_argument("--train", type = str, default = './data/train.csv')
parser.add_argument("--val", type = str, default = './data/valid.csv')
parser.add_argument("--test", type = str, default = './data/test.csv')

args = parser.parse_args() 

lr= args.lr
batch_size= args.batch_size
init = args.init
save_dir = args.save_dir
epochs = args.epochs
dataAugment = args.dataAugment
train_path= args.train
val_path = args.val
test_path = args.test

train = pd.read_csv(train_path)
valid = pd.read_csv(val_path)
test= pd.read_csv(test_path)



def make_arr(df):
  arr= np.array(df)
  return arr[:,1:]

def show_images(image1, image2):
  fig, axes = plt.subplots(nrows=1, ncols=2)
  ax = axes.ravel()
  ax[0].imshow(image1, cmap='gray')
  ax[1].imshow(image2, cmap='gray')

def one_hot(y):
    m= np.shape(y)[0]
    n = 20
    y_one_hot = np.zeros((m,n))
    for i in range(m):
        y_one_hot[i, int(y[i])] = 1
    return y_one_hot

def data_split(data, test_data =False ):
    if test_data == False:
      data_x = np.array(data[:,:-1])
      data_x = np.float32(data_x)
      data_x = data_x.reshape(-1,64,64,3)
      data_y = np.array(data[:,-1])   
      data_y = data_y[:, np.newaxis]
      data_y = one_hot(data_y)
      data_y = np.float32(data_y)
      return data_x, data_y
    
    elif test_data == True:
      data_x = np.array(data[:,:])
      data_x = np.float32(data_x)
      data_x = data_x.reshape(-1,64,64,3)
      return data_x

def add_weights(shape, init=1, name= 'def_name'):
    if init ==1: 
        return tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False, seed = 1234, dtype= tf.float32)(shape), name = name)
    elif init==2:
        return tf.Variable(tf.keras.initializers.he_normal(seed=1234)(shape), name = name)

def add_biases(shape, name= 'def_name'):
    return tf.Variable(tf.constant(0.05, shape=shape), name = name)

def conv_layer(input_layer, kernel, biases, stride_size, padding_type, name= 'def_name'):
     #kernel = add_weights([kernel_size, kernel_size, input_depth, output_depth], init)
     #biases = add_biases([output_depth])
     stride = [1, stride_size, stride_size, 1]
     layer = tf.nn.conv2d(input_layer, kernel, strides=stride, padding= padding_type, name= name) + biases               
     return layer

def pooling_layer(input_layer, kernel_size, stride_size, padding_type, name = 'def_name'):
        kernel = [1, kernel_size, kernel_size, 1]
        stride = [1, stride_size, stride_size, 1]
        return tf.nn.max_pool(input_layer, ksize=kernel, strides=stride, padding=padding_type, name= name)

def flattening_layer(input_layer):
        input_size = input_layer.get_shape().as_list()
        new_size = input_size[-1] * input_size[-2] * input_size[-3]
        return tf.reshape(input_layer, [-1, new_size]),new_size

def fully_connected_layer(input_layer, weights, biases, name= 'def_name'):
        #weights = add_weights([input_shape, output_shape], init)
        #biases = add_biases([output_shape])
        layer = tf.matmul(input_layer,weights , name = name) + biases  # mX+b
        return layer

def activation_layer(layer,  name=''):
     return tf.nn.relu(layer, name= name)





def conv_net(input_layer, init, keep_prob): 
  
    W_Conv1 = add_weights(shape= [5,5,3,32], init = init, name= 'W_Conv1')
    B_Conv1 = add_biases([32])
    Conv1 = conv_layer(input_layer, W_Conv1, B_Conv1, stride_size= 1, padding_type= 'SAME', name= 'Conv1_pre')
    Conv1 = activation_layer(Conv1, name ='Conv1')
    Conv1 = tf.contrib.layers.batch_norm(Conv1 )
    
    W_Conv2 = add_weights(shape= [5,5,32,32], init = init, name= 'W_Conv2')
    B_Conv2 = add_biases([32], name = 'B_Conv2')
    Conv2 = conv_layer(Conv1, W_Conv2, B_Conv2, stride_size= 1, padding_type= 'SAME', name= 'Conv2_pre')   
    Conv2 = activation_layer(Conv2,name ='Conv2')
    Conv2 = tf.contrib.layers.batch_norm(Conv2)
    
    Pool1 = pooling_layer(Conv2, kernel_size= 2, stride_size=2, padding_type='SAME',name ='Pool1')
    
    W_Conv3 = add_weights(shape= [3,3,32,64], init = init, name= 'W_Conv3')
    B_Conv3 = add_biases([64], name = 'B_Conv3')
    Conv3 = conv_layer(Pool1, W_Conv3, B_Conv3, stride_size= 1, padding_type= 'SAME', name= 'Conv3_pre')
    Conv3 = activation_layer(Conv3, name ='Conv3')
    Conv3 = tf.contrib.layers.batch_norm(Conv3)
    
    W_Conv4 = add_weights(shape= [3,3,64,64], init = init, name= 'W_Conv4')
    B_Conv4 = add_biases([64], name = 'B_Conv4')
    Conv4 = conv_layer(Conv3, W_Conv4, B_Conv4, stride_size= 1, padding_type= 'SAME', name= 'Conv4_pre')
    Conv4 = activation_layer(Conv4, name ='Conv4')
    Conv4 = tf.contrib.layers.batch_norm(Conv4)
    
    Pool2 = pooling_layer(Conv4, kernel_size= 2, stride_size=2, padding_type='SAME',name ='Pool2')
    
    W_Conv5 = add_weights(shape= [3,3,64,64], init = init, name= 'W_Conv5')
    B_Conv5 = add_biases([64], name = 'B_Conv5')
    Conv5 = conv_layer(Pool2, W_Conv5, B_Conv5, stride_size= 1, padding_type= 'SAME', name= 'Conv5_pre')
    Conv5 = activation_layer(Conv5, name ='Conv5')
    Conv5 = tf.contrib.layers.batch_norm(Conv5)
    
    W_Conv6 = add_weights(shape= [3,3,64,128], init = init, name= 'W_Conv6')
    B_Conv6 = add_biases([128], name = 'B_Conv6')
    Conv6 = conv_layer(Conv5, W_Conv6, B_Conv6, stride_size= 1, padding_type= 'VALID', name= 'Conv6_pre')
    Conv6 = activation_layer(Conv6, name ='Conv6')
    Conv6 = tf.contrib.layers.batch_norm(Conv6)

    Pool3 = pooling_layer(Conv6, kernel_size= 2, stride_size=2, padding_type='SAME',name ='Pool3')
    
    flattened_layer ,features =flattening_layer(Pool3)
    
    W_FC1 = add_weights(shape= [features,256], init = init, name= 'W_FC1')
    B_FC1 = add_biases([256], name= 'B_FC1')
    FC1 = fully_connected_layer(flattened_layer, W_FC1, B_FC1, name= 'FC1_pre')
    FC1 = activation_layer(FC1, name ='FC1')
    FC1 = tf.nn.dropout(FC1, keep_prob,name ='FC1_drop')
    FC1 = tf.contrib.layers.batch_norm(FC1)
    
    W_FC2 = add_weights(shape= [256,20], init = init, name= 'W_FC2')
    B_FC2 = add_biases([20], name= 'B_FC1')
    FC2 = fully_connected_layer(FC1, W_FC2, B_FC2 ,name ='FC2')
    FC2 = tf.contrib.layers.batch_norm(FC2)
    
    return FC2



def data_augment(data, type_aug, im):
  x = data[:, :-1]
  y = data[:, -1:]
  x = x.reshape(-1, 64,64,3)
  
  if 'horizontal_flip' in type_aug:
    x_aug = x[:,:, ::-1]
    show_images(x[im,:], x_aug[im,:])
    x_aug = x_aug.reshape(-1, 12288)
    data_aug = np.concatenate((x_aug, y), axis =1)
    #data_aug = data_aug[np.random.randint(data_aug.shape[0], size=2000), :]
    data = np.concatenate((data, data_aug), axis=0)
    
  if 'vertical_flip' in type_aug:
    x_aug = x[:, ::-1]
    show_images(x[im,:], x_aug[im,:])
    x_aug = x_aug.reshape(-1, 12288)
    data_aug = np.concatenate((x_aug, y), axis =1)
    #data_aug = data_aug[np.random.randint(data_aug.shape[0], size=2000), :]
    data = np.concatenate((data, data_aug), axis=0)
    
  if 'noise' in type_aug:
    noised= []
    for i in x:
      noised.append(sk.util.random_noise(i))
    x_aug = np.array(noised)
    show_images(x[im,:], x_aug[im,:])
    x_aug = x_aug.reshape(-1, 12288)
    data_aug = np.concatenate((x_aug, y), axis =1)
    #data_aug = data_aug[np.random.randint(data_aug.shape[0], size=2000), :]
    data = np.concatenate((data, data_aug), axis=0)
  
  if 'blur' in type_aug:
    blurred =[]
    for i in x:
      blurred.append(gaussian_filter(i, sigma=2))
    x_aug = np.array(blurred)
    show_images(x[im,:], x_aug[im,:])
    x_aug = x_aug.reshape(-1, 12288)
    data_aug = np.concatenate((x_aug, y), axis =1)
   # data_aug = data_aug[np.random.randint(data_aug.shape[0], size=2000), :]
    data = np.concatenate((data, data_aug), axis=0)

  np.random.shuffle(data)
  return data

# =============================================================================
# =============================================================================
# train = train
# valid= valid
# test = test
# lr= 0.001
# batch_size= 64
# init=2
# save_dir= './drive/My Drive/data2/Output/'
# epochs=12
# dataAugment=1
# =============================================================================
# =============================================================================

train = make_arr(train)
valid = make_arr(valid)
test = make_arr(test)

if dataAugment== 1:
  train = data_augment(train, ['horizontal_flip',  'vertical_flip'], 6)

train.shape

with tf.Session() as sess:

    number_of_classes = len(np.unique(np.array(train[:,-1])))

    images = tf.placeholder(tf.float32,shape=[None,64,64,3], name = 'images')
    labels = tf.placeholder(tf.float32,shape=[None,number_of_classes], name = 'labels')
    keep_prob = tf.placeholder(tf.float32, name= 'keep_prob')
    
    
    out= conv_net(images, init,  keep_prob)
   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out , labels=labels), name = 'cost')
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    score = tf.equal(tf.argmax(out, 1), tf.argmax(labels, 1)) 
    accuracy = tf.reduce_mean(tf.cast(score, tf.float32))
    pred_labels = tf.argmax(out, 1)
            
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
   
    sess.run(tf.global_variables_initializer()) 

    train_X, train_y = data_split(train, test_data = False)
    valid_X, valid_y=  data_split(valid, test_data = False)    
    test_X = data_split(test, test_data = True)
    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    
    
    
    patience = 0
    counter = 0
    for i in range(epochs):
        epoch = i+1
        avg_training_loss= 0
        avg_training_acc = 0
        np.random.shuffle(train)
        train_X, train_y = data_split(train)

        for batch in range(len(train_X)//batch_size):
            counter+=1
            batch_X = train_X[batch*batch_size:min((batch+1)*batch_size,len(train_X))]
            batch_y = train_y[batch*batch_size:min((batch+1)*batch_size,len(train_y))]   
            opt = sess.run(optimizer, feed_dict={images: batch_X, labels: batch_y, keep_prob: 0.5})
            loss, acc = sess.run([cost, accuracy], feed_dict={images: batch_X, labels: batch_y, keep_prob: 0.5})
            avg_training_loss += loss / (len(train_X)//batch_size)
            avg_training_acc += acc / (len(train_X)//batch_size)
            
        print("Iter " + str(epoch) + ", Training Loss= " + \
                      "{:.6f}".format(avg_training_loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(avg_training_acc))
        
        v_loss, v_acc = sess.run([cost, accuracy], feed_dict={images: valid_X, labels : valid_y, keep_prob: 1.0})
     
        print("Iter " + str(epoch) + ", Validation Loss= " + \
                      "{:.6f}".format(v_loss) + ", Validation Accuracy= " + \
                      "{:.5f}".format(v_acc))
        print('\n')
        
        train_loss.append(avg_training_loss)
        valid_loss.append(v_loss)
        train_accuracy.append(avg_training_acc)
        valid_accuracy.append(v_acc)
        
        ## Early Stopping 
        if epoch == 1:
          best_loss = v_loss
          saver = tf.train.Saver( max_to_keep=100)
          saver.save(sess, save_dir + 'saver')
          
        else:
          if v_loss> best_loss:
            patience+=1
            #saver = tf.train.Saver( max_to_keep=1)
            saver.save(sess, save_dir + 'saver')
          else:
            patience =0
            best_loss = v_loss
            #saver = tf.train.Saver( max_to_keep=1)
            saver.save(sess, save_dir + 'saver')
            
        if patience == 5:
          print('Early Stopping')
          print('\n')
          break 
    
    # Saving Filters(Weights) for Conv Layer1
    new_saver = tf.train.Saver()
    with tf.Session() as sess:

      new_saver = tf.train.import_meta_graph(save_dir + 'saver.meta')
      new_saver.restore(sess, tf.train.latest_checkpoint(save_dir))
      
      print('Model Restored')
      with tf.variable_scope('out', reuse=tf.AUTO_REUSE):
          W_Conv1 = tf.get_default_graph().get_tensor_by_name("W_Conv1:0")
          #W_Conv1 = tf.get_variable('W_Conv1', shape=[5, 5, 3, 32])
          weights = W_Conv1.eval(session =sess)
          with open(save_dir + 'Conv1.weights.npz', "wb") as outfile:
              np.save(outfile, weights)

      pred_test = sess.run([pred_labels], feed_dict={images: test_X, keep_prob: 1.0})
      pred_test = np.array(pred_test)
      pred_test = np.transpose(pred_test)

      prediction_df = pd.DataFrame(pred_test)
      prediction_df.index.names = ['id']
      prediction_df.columns = ['label']
      prediction_df.to_csv(save_dir + 'prediction_12.csv')


      summary_writer.close()
      sess.close()



plt.plot(range(1,len(train_loss)+1) ,   train_loss, label = 'training_loss')
plt.plot(range(1,len(valid_loss)+1) ,   valid_loss, label = 'validation_loss')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')

train_loss

## Plotting of 32 layer 1 filters

import numpy as np
import matplotlib.pyplot as plt
weights = np.load(save_dir+ "Conv1.weights.npz")

x_min= np.amin(weights) 
x_max = np.amax(weights)
weights_0_to_1 = (weights - x_min) / (x_max - x_min)
weights_0_to_255_uint8 = (weights_0_to_1)

extracted_filter = (weights_0_to_255_uint8)

#extracted_filter  = weights
fig, ax = plt.subplots(nrows=4, ncols=8)
i=0
for row in ax:
    for col in row:
        col.imshow(extracted_filter[:, :,  0 , i], interpolation='nearest', cmap='seismic')
        col.set_xticks([])
        col.set_yticks([])
        i+=1


tf.reset_default_graph()

#guided backpropagation
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros(tf.shape(grad)))

with tf.Session() as sess:

    g = tf.get_default_graph()
    #saver = tf.train.Saver()
    with g.gradient_override_map({'Relu': 'GuidedRelu'}):
        new_saver = tf.train.import_meta_graph(save_dir + 'saver' + '.meta')

        new_saver.restore(sess, tf.train.latest_checkpoint(save_dir))

    with tf.variable_scope('out', reuse=tf.AUTO_REUSE):
            Conv6 = tf.get_default_graph().get_tensor_by_name("Conv6:0")
            #W_Conv1 = tf.get_variable('W_Conv1', shape=[5, 5, 3, 32])
    images = tf.get_default_graph().get_tensor_by_name("images:0")                

            
    imgs = [train_X[np.argmax(train_y, axis=1) == i][5] for i in range(20)]
    
    grad = [tf.gradients(Conv6[:,:,:,i], images)[0] for i in range(20)]
    
    features = [sess.run(grad[i], feed_dict={images: imgs[i][None]}) for i in range(20)]



x_min= np.amin(imgs) 
x_max = np.amax(imgs)
imgs = (imgs - x_min) / (x_max - x_min)

x_min= np.amin(features) 
x_max = np.amax(features)
features = (features - x_min) / (x_max - x_min)

plt.figure(figsize=(15,15))
for i in range(10):
    plt.subplot(10, 4, 4 * i + 1)
    plt.imshow(np.reshape(imgs[2 * i], [64, 64,3]),interpolation='none')
    
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    
    plt.subplot(10, 4, 4 * i + 2)
    plt.imshow(np.reshape(features[2 * i], [64, 64,3]), interpolation='none')
    
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.subplot(10, 4, 4 * i + 3)
    plt.imshow(np.reshape(imgs[2 * i + 1], [64, 64,3]),interpolation='none')
   
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.subplot(10, 4, 4 * i + 4)
    plt.imshow(np.reshape(features[2 * i + 1], [64, 64,3]),interpolation='none')
   
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

plt.tight_layout()



plt.figure(figsize= (15,15))
plt.subplot(1, 2, 1)
plt.imshow(imgs[17] ,interpolation='none')
plt.subplot(1, 2, 2)
plt.imshow(features[17, 0, :] ,interpolation='none')