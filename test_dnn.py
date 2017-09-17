import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import random

def create_train_and_test_data():
  _data = pd.read_csv('processed_data.csv', names = ["class", "enemies", "health", "distance"])
  X = _data.drop('class',axis=1)
  y = _data['class']

  fee = []
  for index, row in _data.iterrows():
    tempArr = row.as_matrix(columns=None)
    tArrSec = np.array([tempArr[1],tempArr[2],tempArr[3]])
    tArrSec = list(tArrSec)
    if tempArr[0] == 1:
      fee.append([tArrSec, [1,0]])
    else:
      fee.append([tArrSec, [0,1]])
    # print(fee)
    # break
  # random.shuffle(fee)
  fee = np.array(fee)
  testing_size = int(0.1 * len(_data))

  train_x = list(fee[:,0][:-testing_size])
  train_y = list(fee[:,1][:-testing_size])
  
  test_x = list(fee[:,0][-testing_size:])
  test_y = list(fee[:,1][-testing_size:])
  # print(test_y)
  # print(train_y)
  return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = create_train_and_test_data()

# defining the model
n_nodes_h1 = 500
n_nodes_h2 = 500
n_nodes_h3 = 500

n_classes = 2
# number of feature batches that is going to go through the network
batch_size = 100

x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')


def neural_network_model(data):
  hidden1 =  { 'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_h1])), 'biases':tf.Variable(tf.random_normal([n_nodes_h1]))}
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

# saver object to save the model

tf_log = 'tf.log'

def train_neural_network(x):
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
  optimizer = tf.train.AdamOptimizer().minimize(cost)

  hm_epochs = 3
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    try:
      epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
      print('STARTING:',epoch)
    except:
      epoch = 1

    # for epoch in range(hm_epochs):
    #   epoch_loss = 0
    while epoch <= hm_epochs:

      if epoch != 1:
          saver.restore(sess,"./model/model.ckpt")
      epoch_loss = 1
      # for _ in range(int(mnist.train.num_examples/batch_size)):
      #   e_x, e_y = mnist.train.next_batch(batch_size) 
      i = 0
      while i < len(train_x):
        start = i
        end = i+batch_size
        batch_x = np.array(train_x[start:end])
        batch_y = np.array(train_y[start:end])

        _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
        epoch_loss += c
        i += batch_size
      
      saver.save(sess, "./model/model.ckpt")
      print('Epoch', epoch, 'completed out of', hm_epochs, 'loss: ', epoch_loss)
      with open(tf_log,'a') as f:
        f.write(str(epoch)+'\n') 
      epoch +=1

    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print(accuracy.eval({x:test_x, y:test_y}))
    
train_neural_network(x)

# if _name_ == '_main_':
# train_x, train_y, test_x, test_y = create_train_and_test_data()
# with open('split_data.pickle', 'wb') as f:
#   print('dumping')
#   pickle.dump([train_x, train_y, test_x, test_y], f)

