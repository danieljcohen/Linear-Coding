import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 3.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    #calculating range var:
    high = (6**(1/2))/((in_size+out_size)**(1/2))
    low = -1*high

    #initializing W
    W = np.random.uniform(low=low,high=high,size=(in_size,out_size))

    #initializing b
    b = np.random.uniform(low=low,high=high,size=(out_size,))

    params['W' + name] = W
    params['b' + name] = b

# Q 3.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = 1/(1+np.exp(-x))
    return res

# Q 3.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    #print('W' + name)
    b = params['b' + name]

    # your code here
    pre_act = X@W+b
    post_act = activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 3.2.2
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    x = x - x.max()
    res = np.exp(x)/np.sum(np.exp(x), axis=1,keepdims = True)
    
    return res

# Q 3.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    loss = ((-1 * y) * np.log(probs))
    loss = np.sum(loss)
    #calculating acc:
    correct_count = np.sum(np.argmax(probs,axis=1)==np.argmax(y,axis=1))
    total_count = y.shape[0]
    acc = correct_count/total_count
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 3.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    grad_a = activation_deriv(post_act)
    #print("DLDZ: ", (delta*grad_a).shape)
    #print("X: ", X.shape)
    #print("W: ", W.shape)
    grad_W = X.T@(delta*grad_a)
    grad_b=(delta*grad_a).T@np.ones((X.shape[0],))
    grad_X = (delta*grad_a)@W.T
    

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 3.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    indices = np.random.permutation(len(x))
    x_shuffled = x[indices]
    y_shuffled = y[indices]
     # Split into batches
    num_batches = len(x) // batch_size
    batches = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_x = x_shuffled[start_idx:end_idx]
        batch_y = y_shuffled[start_idx:end_idx]
        batches.append((batch_x, batch_y))

    if len(x) % batch_size != 0:
        start_idx = num_batches * batch_size
        batch_x = x_shuffled[start_idx:]
        batch_y = y_shuffled[start_idx:]
        batches.append((batch_x, batch_y))
    return batches
