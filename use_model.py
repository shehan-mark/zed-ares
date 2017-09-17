import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

# defining the model
n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

n_classes = 2
# number of feature batches that is going to go through the network
batch_size = 100

x = tf.placeholder('float', [None, 3])
y = tf.placeholder('float')


def neural_network_model(data):
  hidden1 =  { 'weights':tf.Variable(tf.random_normal([3, n_nodes_h1])), 'biases':tf.Variable(tf.random_normal([n_nodes_h1]))}
  # hidden2 =  { 'weights':tf.Variable(tf.random_normal([n_nodes_h1, n_nodes_h2])), 'biases':tf.Variable(tf.random_normal([n_nodes_h2]))}
  # hidden3 =  { 'weights':tf.Variable(tf.random_normal([n_nodes_h2, n_nodes_h3])), 'biases':tf.Variable(tf.random_normal([n_nodes_h3]))}

  output_l =  { 'weights':tf.Variable(tf.random_normal([n_nodes_h1, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

  l1 = tf.add(tf.matmul(data, hidden1['weights']), hidden1['biases'])
  l1 = tf.nn.relu(l1)

  # l2 = tf.add(tf.matmul(l1, hidden2['weights']), hidden2['biases'])
  # l2 = tf.nn.relu(l2)

  # l3 = tf.add(tf.matmul(l2, hidden3['weights']), hidden3['biases'])
  # l3 = tf.nn.relu(l3)

  output = tf.matmul(l1, output_l['weights']) + output_l['biases']

  return output



def use_neural_network():
  prediction = neural_network_model(x)
  # _data = pd.read_csv('processed_data.csv', names = ["class", "enemies", "health", "distance"])
  # X = _data.drop('class',axis=1)    
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess,"./3 - model/model.ckpt")
    
    inputA = [2,100,500]
    result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[inputA]}),1)))
    preds = prediction.eval(feed_dict={x:[inputA]})
    print(preds)
    print(result)
    if preds[0][0] < preds[0][1]:
      print('aggressive')
    else:
      print('normal')

use_neural_network()