
# coding: utf-8

# In[9]:


import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt
from tf_utils import random_mini_batches


# In[3]:


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32,[n_x,None],name = "X");
    Y = tf.placeholder(tf.float32,[n_y,None],name = "Y");
    return X,Y;


# In[5]:


def initialization_parameters():
    tf.set_random_seed(1);
    W1 = tf.Variable(tf.truncated_normal(shape=[25,12288],stddev = 0.1),name = "W1")
    b1 = tf.Variable(tf.zeros(shape=[25,1]),name = "b1")
    W2 = tf.Variable(tf.truncated_normal(shape=[12,25],stddev = 0.1),name = "W2")
    b2 = tf.Variable(tf.zeros(shape=[12,1]),name = "b2")
    W3 = tf.Variable(tf.truncated_normal(shape=[6,12],stddev = 0.1),name = "W3")
    b3 =tf.Variable(tf.zeros(shape=[6,1]),name = "b3")
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2,
                  "W3":W3,
                  "b3":b3};
    return parameters;


# In[6]:


def forward_prop(X,parameters):
    
    W1 = parameters["W1"];
    b1 = parameters["b1"];
    W2 = parameters["W2"];
    b2 = parameters["b2"];
    W3 = parameters["W3"];
    b3 = parameters["b3"];
    
    Z1 = tf.add(tf.matmul(W1,X),b1);
    A1 = tf.nn.relu(Z1);
    Z2 = tf.add(tf.matmul(W2,A1),b2);
    A2 = tf.nn.relu(Z2);
    Z3 = tf.add(tf.matmul(W3,A2),b3);
    
    return Z3;


# In[22]:


def cost_function(Z3,Y):
    
    logits = tf.transpose(Z3);
    labels = tf.transpose(Y);
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels));
    return cost;


# In[24]:


def back_prop(cost,learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost);
    return optimizer;


# In[27]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
    costs=[];
    seed = 2;
    ops.reset_default_graph();
    tf.set_random_seed(1);
    (n_x,m)=X_train.shape;
    (n_y,_)=Y_train.shape;
    
    X,Y = create_placeholders(n_x, n_y);
    
    parameters = initialization_parameters()
    
    Z3 = forward_prop(X,parameters);
    
    cost = cost_function(Z3,Y);
    
    optimizer = back_prop(cost,learning_rate);
    
    init = tf.global_variables_initializer();
    
    with tf.Session() as sess:
        
        sess.run(init);
        for epoch in range(num_epochs):
            epoch_cost = 0;
            num_minibatchs = int(m/minibatch_size);
            seed +=1;
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            for minibatch in minibatches:
                (miniBatch_X,miniBatch_Y) = minibatch;
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict = {X:miniBatch_X,Y:miniBatch_Y});
                epoch_cost += minibatch_cost / num_minibatchs;
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters);
        
        print ("Parameters have been trained!")
        
        correct_prediction = tf.equal(tf.argmax(Z3,axis = 0), tf.argmax(Y,axis=0))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters;
        


# In[28]:




