
# coding: utf-8

# In[1]:


import tensorflow as tf
import math

NUM_CLASSES=10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def initialization_put():
    
    n_x = 784;
    n_y = 10;
    
    #X = tf.placeholder(tf.float32,shape = [n_x,None],name = "X");
    #Y = tf.placeholder(tf.float32,shape = [n_y,None],name = "Y");
    X = tf.placeholder(tf.float32,shape = [None,784],name = "X");
    Y = tf.placeholder(tf.float32,shape = [None,10],name = "Y");
    return X,Y;
    


# In[6]:


def forward_prop(images,hidden1_units,hidden2_units):
    """tf.set_random_seed(1) 
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
                tf.truncated_normal([hidden1_units, IMAGE_PIXELS],mean=0.0,
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                                 name='weights')
        biases = tf.Variable(tf.zeros((hidden1_units,1)),
                         name='biases')
        hidden1 = tf.nn.relu(tf.add(tf.matmul(weights, images) , biases))
    with tf.name_scope("hidden2"):
        weights =  tf.Variable(tf.truncated_normal([hidden2_units,hidden1_units],mean=0.0,stddev=1.0 / math.sqrt(float(hidden1_units))),name = "weights");
        
        bias = tf.Variable(tf.zeros((hidden2_units,1)),name = "bias");
        
        hidden2 = tf.nn.relu(tf.add(tf.matmul(weights,hidden1),bias));
    with tf.name_scope("softmaxlinear"):
        weights = tf.Variable(tf.truncated_normal([NUM_CLASSES,hidden2_units],mean=0.0,stddev=1.0 / math.sqrt(float(hidden2_units))),name = "weights");
        bias = tf.Variable(tf.zeros((NUM_CLASSES,1)),name = "bias");
       """ 
    #W = tf.Variable(tf.truncated_normal([784,10],mean=0.0,stddev=1.0 / math.sqrt(float(784))),name = "weights")
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

    
        


# In[8]:


def cost(logits,labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
    return tf.reduce_mean(cross_entropy)


# In[9]:


def optimizer_cost(cost,learning_rate):
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #train_op = optimizer.minimize(cost, global_step=global_step)
    train_op = optimizer.minimize(cost)
    return train_op;


# In[11]:


def predict_accuary(y_pred,y):
    current_predict = tf.equal(tf.argmax(y), tf.argmax(y_pred));
    accuracy = tf.reduce_mean(tf.cast(current_predict,tf.float32))
    return accuracy;

