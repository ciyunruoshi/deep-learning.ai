
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


# In[23]:


def convNetNN_forward_prop(input_train):
    with tf.name_scope("input"):
        X = tf.reshape(input_train,shape = [-1,28,28,1]);
    with tf.name_scope("conv1"):
        W = weight_init(shape = [5,5,1,32]);
        b = bias([32]);
        conv1_output = tf.nn.relu(conv2d(X,W)+b);
    with tf.name_scope("pool1"):
        pool1_output = max_pool(conv1_output);
    with tf.name_scope("conv2"):
        W = weight_init(shape = [5,5,32,64]);
        b = bias([64])
        conv2_output = tf.nn.relu(conv2d(pool1_output,W)+b);
    with tf.name_scope("pool2"):
        pool2_output  = max_pool(conv2_output);
    with tf.name_scope("fc1"):
        pool_flat = tf.reshape(pool2_output,shape = [-1,7*7*64]);
        W = weight_init(shape = [7*7*64,1024]);
        b = bias([1024])
        fc1_output = tf.nn.relu(tf.matmul(pool_flat,W)+b);
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32)
        dropout_ouput = tf.nn.dropout(fc1_output, keep_prob);
        
    with tf.name_scope("fc2"):
        W = weight_init(shape = [1024,10]);
        b = bias([10])
        fc2_output = tf.nn.relu(tf.matmul(dropout_ouput,W)+b);
    return fc2_output,keep_prob;


# In[24]:


def weight_init(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1));
    return w;


# In[25]:


def bias(shape):
    initial = tf.constant(0.1, shape=shape);
    return tf.Variable(initial);


# In[26]:


def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME');


# In[27]:


def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = "SAME");


# In[32]:


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True);
    num = FLAGS.num;
    
    x = tf.placeholder(tf.float32,shape = [None,784]);
    y = tf.placeholder(tf.float32,shape = [None,10]);
    
    y_predict,keep_prob =  convNetNN_forward_prop(x);
    
    with tf.name_scope("loss"):
        loss_batch = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict);
        loss = tf.reduce_mean(loss_batch);
    with tf.name_scope("optimizer"):
        optimizer =  tf.train.AdamOptimizer(1e-4).minimize(loss)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_predict, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        for i in range(num):
            batch = mnist.train.next_batch(50)
            sess.run(optimizer,feed_dict={x:batch[0],y:batch[1],keep_prob:0.5});
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy));
if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--data_dir",type = str,default='D:\MNIST_data',help='Directory for storing input data');
    parser.add_argument("--num",type=int,default=20000);
    FLAGS,unparsed = parser.parse_known_args();
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed);

        

