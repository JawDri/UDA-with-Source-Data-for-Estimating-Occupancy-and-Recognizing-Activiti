#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#
from __future__ import print_function

import numpy as np
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
from tf_slim.layers import layers as _layers
#from tensorflow.contrib.layers import fully_connected
#from tensorflow.contrib.framework import get_variables
from tensorflow.python.ops import math_ops, array_ops, random_ops, nn_ops
import matplotlib.pyplot as plt
import imageio
import platform
if platform.system() == 'Darwin':
    import matplotlib
    matplotlib.use('TkAgg')
tf.compat.v1.disable_eager_execution()


def get_variables(scope=None,
                  suffix=None,
                  collection=tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
  """Gets the list of variables, filtered by scope and/or suffix.
  Args:
    scope: an optional scope for filtering the variables to return. Can be a
      variable scope or a string.
    suffix: an optional suffix for filtering the variables to return.
    collection: in which collection search for. Defaults to
      `GraphKeys.GLOBAL_VARIABLES`.
  Returns:
    a list of variables in collection with scope and suffix.
  """
  if isinstance(scope, tf.compat.v1.VariableScope):
    scope = scope.name
  if suffix is not None:
    if ':' not in suffix:
      suffix += ':'
    scope = (scope or '') + '.*' + suffix
  return tf.compat.v1.get_collection(collection, scope)

def toyNet(X):
    # Define network architecture
    with tf.compat.v1.variable_scope('Generator'):
        net = _layers.fully_connected(X, 15, activation_fn=tf.nn.relu)
        net = _layers.fully_connected(net, 15, activation_fn=tf.nn.relu)
        net = _layers.fully_connected(net, 15, activation_fn=tf.nn.relu)
        
    with tf.compat.v1.variable_scope('Classifier1'):
        net1 = _layers.fully_connected(net, 15, activation_fn=tf.nn.relu)
        net1 = _layers.fully_connected(net1, 15, activation_fn=tf.nn.relu)
        net1 = _layers.fully_connected(net1, 2, activation_fn=None)
        logits1 = tf.sigmoid(net1)
    with tf.compat.v1.variable_scope('Classifier2'):
        net2 = _layers.fully_connected(net, 15, activation_fn=tf.nn.relu)
        net2 = _layers.fully_connected(net2, 15, activation_fn=tf.nn.relu)
        net2 = _layers.fully_connected(net2, 2, activation_fn=None)
        logits2 = tf.sigmoid(net2)
    return logits1, logits2


def sort_rows(matrix, num_rows):
    matrix_T = array_ops.transpose(matrix, [1, 0])
    sorted_matrix_T = nn_ops.top_k(matrix_T, num_rows)[0]
    return array_ops.transpose(sorted_matrix_T, [1, 0])


def discrepancy_slice_wasserstein(p1, p2):
    s = array_ops.shape(p1)
    if p1.get_shape().as_list()[1] > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        proj = random_ops.random_normal([array_ops.shape(p1)[1], 128])
        proj *= math_ops.rsqrt(math_ops.reduce_sum(math_ops.square(proj), 0))#, keep_dims=True
        p1 = math_ops.matmul(p1, proj)
        p2 = math_ops.matmul(p2, proj)
    p1 = sort_rows(p1, s[0])
    p2 = sort_rows(p2, s[0])
    wdist = math_ops.reduce_mean(math_ops.square(p1 - p2))
    return math_ops.reduce_mean(wdist)


def discrepancy_mcd(out1, out2):
    return tf.reduce_mean(input_tensor=tf.abs(out1 - out2))


def load_data():
    # Load inter twinning moons 2D dataset by F. Pedregosa et al. in JMLR 2011
    '''moon_data = np.load('moon_data.npz')
    x_s = moon_data['x_s']
    
    y_s = moon_data['y_s']
    
    x_t = moon_data['x_t']'''
    
    Source_train = pd.read_csv("/content/drive/MyDrive/SWD/data/Source_train.csv")
    Source_test = pd.read_csv("/content/drive/MyDrive/SWD/data/Source_test.csv")
    Target_train = pd.read_csv("/content/drive/MyDrive/SWD/data/Target_train.csv")
    Target_test = pd.read_csv("/content/drive/MyDrive/SWD/data/Target_test.csv")
    FEATURES = list(i for i in Source_train.columns if i!= 'labels')
    TARGET = "labels"
    
    x_s = np.array(Source_train[FEATURES])
    y_s = np.array(Source_train[TARGET]).reshape(len(Source_train[TARGET]),1)
    x_t = np.array(Target_train[FEATURES])
    y_t = np.array(Target_train[TARGET]).reshape(len(Target_train[TARGET]),1)
  
    x_t_t = np.array(Target_test[FEATURES])
    y_t_t = np.array(Target_test[TARGET]).reshape(len(Target_test[TARGET]),1)

    return x_s, y_s, x_t, y_t, x_t_t, y_t_t


def generate_grid_point():
    x_min, x_max = x_s[:, 0].min() - .5, x_s[:, 0].max() + 0.5
    y_min, y_max = x_s[:, 1].min() - .5, x_s[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    return xx, yy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default="adapt_swd",
                        choices=["source_only", "adapt_mcd", "adapt_swd"])
    opts = parser.parse_args()

    # Load data
    x_s, y_s, x_t, y_t, x_t_t, y_t_t = load_data()
    
    

    # set random seed
    tf.compat.v1.set_random_seed(1235)

    # Define TF placeholders
    X = tf.compat.v1.placeholder(tf.float32, shape=[None, x_s.shape[1]])
    Y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
    X_target = tf.compat.v1.placeholder(tf.float32, shape=[None, x_t.shape[1]])
    X_target_t = tf.compat.v1.placeholder(tf.float32, shape=[None, x_t_t.shape[1]])

    # Network definition
    with tf.compat.v1.variable_scope('toyNet'):
        logits1, logits2 = toyNet(X)
    with tf.compat.v1.variable_scope('toyNet', reuse=True):
        logits1_target, logits2_target = toyNet(X_target)

    with tf.compat.v1.variable_scope('toyNet', reuse=True):
        logits1_target_t, logits2_target_t = toyNet(X_target_t)

    # Cost functions
    eps = 1e-05
    cost1 = -tf.reduce_mean(input_tensor=Y * tf.math.log(logits1 + eps) + (1 - Y) * tf.math.log(1 - logits1 + eps))
    cost2 = -tf.reduce_mean(input_tensor=Y * tf.math.log(logits2 + eps) + (1 - Y) * tf.math.log(1 - logits2 + eps))
    loss_s = cost1 + cost2

    if opts.mode == 'adapt_swd':
        loss_dis = discrepancy_slice_wasserstein(logits1_target, logits2_target)
    else:
        loss_dis = discrepancy_mcd(logits1_target, logits2_target)
    
    # Setup optimizers
    variables_all = get_variables(scope='toyNet')
    variables_generator = get_variables(scope='toyNet' + '/Generator')
    variables_classifier1 = get_variables(scope='toyNet' + '/Classifier1')
    variables_classifier2 = get_variables(scope='toyNet' + '/Classifier2')

    optim_s = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_s, var_list=variables_all)
    optim_dis1 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_s - loss_dis, var_list=variables_classifier1)
    optim_dis2 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_s - loss_dis, var_list=variables_classifier2)
    optim_dis3 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.005).\
        minimize(loss_dis, var_list=variables_generator)
    
    # Select predictions from C1
    predicted1 = tf.cast(logits2_target_t > 0.5, dtype=tf.float32)
    
    # Generate grid points for visualization
    #xx, yy = generate_grid_point()
    
    # For creating GIF purpose
    gif_images = []

    #Evaluate model
    correct_pred = tf.equal(predicted1, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    TP = tf.compat.v1.count_nonzero(predicted1 * Y)
    TN = tf.compat.v1.count_nonzero((predicted1 - 1) * (Y - 1))
    FP = tf.compat.v1.count_nonzero(predicted1 * (Y - 1))
    FN = tf.compat.v1.count_nonzero((predicted1 - 1) * Y)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    #confusion_mat = tf.math.confusion_matrix(tf.argmax(Y, 1), tf.argmax(predicted1, 0))
    
    
    # Start session
    with tf.compat.v1.Session() as sess:
        if opts.mode == 'source_only':
            print('-> Perform source only training. No adaptation.')
            train = optim_s
        else:
            print('-> Perform training with domain adaptation.')
            train = tf.group(optim_s, optim_dis1, optim_dis2, optim_dis3)

        # Initialize variables
        net_variables = tf.compat.v1.global_variables() + tf.compat.v1.local_variables()
        sess.run(tf.compat.v1.variables_initializer(net_variables))

        # Training
        for step in range(100001):
            if step % 1000 == 0:
                print("Iteration: %d / %d" % (step, 10000))
                '''
                Z = sess.run(predicted1, feed_dict={X: np.c_[xx.ravel(), yy.ravel()]})
                Z = Z.reshape(xx.shape)
                f = plt.figure()
                plt.contourf(xx, yy, Z, cmap=plt.cm.copper_r, alpha=0.9)
                plt.scatter(x_s[:, 0], x_s[:, 1], c=y_s.reshape((len(x_s))),
                            cmap=plt.cm.coolwarm, alpha=0.8)
                plt.scatter(x_t[:, 0], x_t[:, 1], color='green', alpha=0.7)
                plt.text(1.6, -0.9, 'Iter: ' + str(step), fontsize=14, color='#FFD700',
                         bbox=dict(facecolor='dimgray', alpha=0.7))
                plt.axis('off')
                f.savefig(opts.mode + '_iter' + str(step) + ".png", bbox_inches='tight',
                          pad_inches=0, dpi=100, transparent=True)
                gif_images.append(imageio.imread(
                                  opts.mode + '_iter' + str(step) + ".png"))
                plt.close()'''

                # Calculate accuracy for 256 mnist test images                
                print("Testing Accuracy:", \
                  sess.run(accuracy, feed_dict={X_target_t: x_t_t,
                                      Y:y_t_t}))
                print("Testing F1-score:", \
                  sess.run(f1, feed_dict={X_target_t: x_t_t,
                                      Y:y_t_t})) 
                '''print("Testing F1-score-labels:", \
                  sess.run(confusion_mat, feed_dict={X_target_t: x_t_t,
                                      Y:y_t_t}))   '''                                                        
                                      
            # Forward and backward propagation
            _ = sess.run([train], feed_dict={X: x_s, Y: y_s, X_target: x_t})
            

        '''# Save GIF
        imageio.mimsave(opts.mode + '.gif', gif_images, duration=0.8)
        print("[Finished]\n-> Please see the current folder for outputs.")'''
