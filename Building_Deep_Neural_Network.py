
# coding: utf-8

### Building Deep Neural Network ###

#
# Goal:
# - Build a deep neural network with many layers.
# - Will implement all the functions required to build a deep neural network.
# - Use non-linear units like ReLU to improve model
# - Build a deeper neural network (with more than 1 hidden layer)
# - Implement an easy-to-use neural network class
# 
# **Notation**:
# - Superscript $[l]$ denotes a quantity associated with the $l^{th}$ layer. 
#     - Example: $a^{[L]}$ is the $L^{th}$ layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
# - Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example. 
#     - Example: $x^{(i)}$ is the $i^{th}$ training example.
# - Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
#     - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
#

### 1) Packages 
# 

import numpy as np                   # main package for scientific computing in Python
import h5py                          # helps store huge amounts of numerical data
import matplotlib.pyplot as plt      # a library to plot graphs in Python
from testCases_v4a import *          # provides test cases to assess the correctness of functions used
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1) #used to keep all random function calls consistent


### 2) Outline of What is Being Done
# 
# To build neural network, will be implementing several "helper functions". Here is an outline:
# 
# - Initialize the parameters for a two-layer network and for an $L$-layer neural network.
# - Implement the forward propagation module (shown in purple in the figure below).
#      - Complete the LINEAR part of a layer's forward propagation step (resulting in $Z^{[l]}$).
#      - ACTIVATION function given (relu/sigmoid).
#      - Combine the previous two steps into a new [LINEAR->ACTIVATION] forward function.
#      - Stack the [LINEAR->RELU] forward function L-1 time (for layers 1 through L-1) and add a [LINEAR->SIGMOID] at the end (for the final layer $L$). Gives a new L_model_forward function.
# - Compute the loss.
# - Implement the backward propagation module (denoted below).
#     - Complete the LINEAR part of a layer's backward propagation step.
#     - Gradient of the ACTIVATE function given (relu_backward/sigmoid_backward) 
#     - Combine the previous two steps into a new [LINEAR->ACTIVATION] backward function.
#     - Stack [LINEAR->RELU] backward L-1 times and add [LINEAR->SIGMOID] backward in a new L_model_backward function
# - Finally, update the parameters.
# 
# 

### 3) Initialization
# 
# Write two helper functions that will initialize the parameters for the model.
# The first function will be used to initialize parameters for a two layer model.
# The second one will generalize this initialization process to $L$ layers.
# 
### 3.1) 2-layer Neural Network
# 
# Create and initialize the parameters of the 2-layer neural network.
# 
# Note:
# - The model's structure is: *LINEAR -> RELU -> LINEAR -> SIGMOID*. 
# - Use random initialization for the weight matrices. Use `np.random.randn(shape)*0.01` with the correct shape.
# - Use zero initialization for the biases. Use `np.zeros(shape)`.
#

### initialize_parameters  ###

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    


#

parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 

### 3.2) L-layer Neural Network
# 
# The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors.
# When completing the `initialize_parameters_deep`, make sure that dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l
#

### initialize_parameters_deep ###

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        # Implementation of one-layer network
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 # Random initialization for weight matrices
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1)) # zeros initialization for the biases
        
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


# Initializing deeper L-layer neural network

parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected output**:
#        
# <table style="width:80%">
#   <tr>
#     <td> **W1** </td>
#     <td>[[ 0.01788628  0.0043651   0.00096497 -0.01863493 -0.00277388]
#  [-0.00354759 -0.00082741 -0.00627001 -0.00043818 -0.00477218]
#  [-0.01313865  0.00884622  0.00881318  0.01709573  0.00050034]
#  [-0.00404677 -0.0054536  -0.01546477  0.00982367 -0.01101068]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1** </td>
#     <td>[[ 0.]
#  [ 0.]
#  [ 0.]
#  [ 0.]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2** </td>
#     <td>[[-0.01185047 -0.0020565   0.01486148  0.00236716]
#  [-0.01023785 -0.00712993  0.00625245 -0.00160513]
#  [-0.00768836 -0.00230031  0.00745056  0.01976111]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b2** </td>
#     <td>[[ 0.]
#  [ 0.]
#  [ 0.]]</td> 
#   </tr>
#   
# </table>

### 4) Forward propagation module 

### linear_forward ###

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    # Linear module of forward propagation
    Z = np.dot(W, A) + b
 
    
    assert(Z.shape == (W.shape[0], A.shape[1])) # Returns the dimensions of an array
    cache = (A, W, b)
    
    return Z, cache


# Check specific responses to a particular set of inputs

A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))


# **Expected output**:
# 
# <table style="width:35%">
#   
#   <tr>
#     <td> **Z** </td>
#     <td> [[ 3.26295337 -1.23429987]] </td> 
#   </tr>
#   
# </table>

### 4.2) Linear-Activation Forward
#


### linear_activation_forward ###

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        # Implement forward propagation using sigmoid
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)            # Z will feed into backward function
       
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        # Implement forward propagation using ReLU
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)               # This will feed into backward function
        
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)        #return values of a function, depending on the argument

    return A, cache


# Check specific responses of A_prev, W, and b

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))


# **Expected output**:
#        
# <table style="width:35%">
#   <tr>
#     <td> **With sigmoid: A ** </td>
#     <td > [[ 0.96890023  0.11013289]]</td> 
#   </tr>
#   <tr>
#     <td> **With ReLU: A ** </td>
#     <td > [[ 3.43896131  0.        ]]</td> 
#   </tr>
# </table>
# 

# Note: In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single layer in the neural network, not two layers. 

### d) L-Layer Model 
# 

### L_model_forward ###

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):                     # Use for loop to replicate (L-1) times
        A_prev = A 
        A, cache = linear_activation_forward(A_prev,    
                                            parameters['W' + str(l)],
                                            parameters['b' + str(l)],
                                            activation='relu')
        caches.append(cache)
       
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A,
                                         parameters['W' + str(L)],
                                         parameters['b' + str(L)],
                                         activation='sigmoid')
    caches.append(cache)
    
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches


# 

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# <table style="width:50%">
#   <tr>
#     <td> **AL** </td>
#     <td > [[ 0.03921668  0.70498921  0.19734387  0.04728177]]</td> 
#   </tr>
#   <tr>
#     <td> **Length of caches list ** </td>
#     <td > 3 </td> 
#   </tr>
# </table>
#

### 5) Cost function

# 
# Compute the cross-entropy cost $J$, using the following formula:
# $$-\frac{1}{m} \sum\limits_{i = 1}^{m} (y^{(i)}\log\left(a^{[L] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right)) \tag{7}$$
# 

# 

### compute_cost ###

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    
    
    cost = np.squeeze(cost)      # To make sure the cost's shape is what is expected (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost


# 

Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))


# **Expected Output**:
# 
# <table>
# 
#     <tr>
#     <td>**cost** </td>
#     <td> 0.2797765635793422</td> 
#     </tr>
# </table>

### 6) Backward propagation module
# 
# Now, similar to forward propagation, going to build the backward propagation in three steps:
# - LINEAR backward
# - LINEAR -> ACTIVATION backward where ACTIVATION computes the derivative of either the ReLU or sigmoid activation
# - [LINEAR -> RELU] $\times$ (L-1) -> LINEAR -> SIGMOID backward (whole model)

### 6.1) Linear backward
# 
# For layer $l$, the linear part is: $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$ (followed by an activation).
# 
# Suppose you have already calculated the derivative $dZ^{[l]} = \frac{\partial \mathcal{L} }{\partial Z^{[l]}}$. Want to get $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$. 
# 
# The three outputs $(dW^{[l]}, db^{[l]}, dA^{[l-1]})$ are computed using the input $dZ^{[l]}$.Here are the formulas you need:
# $$ dW^{[l]} = \frac{\partial \mathcal{J} }{\partial W^{[l]}} = \frac{1}{m} dZ^{[l]} A^{[l-1] T} \tag{8}$$
# $$ db^{[l]} = \frac{\partial \mathcal{J} }{\partial b^{[l]}} = \frac{1}{m} \sum_{i = 1}^{m} dZ^{[l](i)}\tag{9}$$
# $$ dA^{[l-1]} = \frac{\partial \mathcal{L} }{\partial A^{[l-1]}} = W^{[l] T} dZ^{[l]} \tag{10}$$
# 

# Use the 3 formulas above to implement linear_backward().

# 

### linear_backward ###

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache                            # tuple of values coming from forward propagation in the current layer
    m = A_prev.shape[1]
    # Implement linear portion of backward propagation (linear_backward)
    dW = np.dot(dZ, cache[0].T) / m 
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(cache[1].T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


#

# Set up some test inputs
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# ** Expected Output**:
#     
# ```
# dA_prev = 
#  [[-1.15171336  0.06718465 -0.3204696   2.09812712]
#  [ 0.60345879 -3.72508701  5.81700741 -3.84326836]
#  [-0.4319552  -1.30987417  1.72354705  0.05070578]
#  [-0.38981415  0.60811244 -1.25938424  1.47191593]
#  [-2.52214926  2.67882552 -0.67947465  1.48119548]]
# dW = 
#  [[ 0.07313866 -0.0976715  -0.87585828  0.73763362  0.00785716]
#  [ 0.85508818  0.37530413 -0.59912655  0.71278189 -0.58931808]
#  [ 0.97913304 -0.24376494 -0.08839671  0.55151192 -0.10290907]]
# db = 
#  [[-0.14713786]
#  [-0.11313155]
#  [-0.13209101]]
# ```

### 6.2) Linear-Activation backward
#

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    # compute relu_backward
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db =  linear_backward(dZ, linear_cache)
        
    # compute sigmoid_backward    
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    
    return dA_prev, dW, db


#

dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# **Expected output with sigmoid:**
# 
# <table style="width:100%">
#   <tr>
#     <td > dA_prev </td> 
#            <td >[[ 0.11017994  0.01105339]
#  [ 0.09466817  0.00949723]
#  [-0.05743092 -0.00576154]] </td> 
# 
#   </tr> 
#   
#     <tr>
#     <td > dW </td> 
#            <td > [[ 0.10266786  0.09778551 -0.01968084]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > db </td> 
#            <td > [[-0.05729622]] </td> 
#   </tr> 
# </table>
# 
# 

# **Expected output with relu:**
# 
# <table style="width:100%">
#   <tr>
#     <td > dA_prev </td> 
#            <td > [[ 0.44090989  0.        ]
#  [ 0.37883606  0.        ]
#  [-0.2298228   0.        ]] </td> 
# 
#   </tr> 
#   
#     <tr>
#     <td > dW </td> 
#            <td > [[ 0.44513824  0.37371418 -0.10478989]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > db </td> 
#            <td > [[-0.20837892]] </td> 
#   </tr> 
# </table>
# 
# 

### 6.3) L-Model Backward 

# 

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    # Implement backwards function for the whole network
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# 

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)


# **Expected Output**
# 
# <table style="width:60%">
#   
#   <tr>
#     <td > dW1 </td> 
#            <td > [[ 0.41010002  0.07807203  0.13798444  0.10502167]
#  [ 0.          0.          0.          0.        ]
#  [ 0.05283652  0.01005865  0.01777766  0.0135308 ]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > db1 </td> 
#            <td > [[-0.22007063]
#  [ 0.        ]
#  [-0.02835349]] </td> 
#   </tr> 
#   
#   <tr>
#   <td > dA1 </td> 
#            <td > [[ 0.12913162 -0.44014127]
#  [-0.14175655  0.48317296]
#  [ 0.01663708 -0.05670698]] </td> 
# 
#   </tr> 
# </table>
# 
# 

### 6.4) Update Parameters
# 
# Will update the parameters of the model, using gradient descent: 
# 
# $$ W^{[l]} = W^{[l]} - \alpha \text{ } dW^{[l]} \tag{16}$$
# $$ b^{[l]} = b^{[l]} - \alpha \text{ } db^{[l]} \tag{17}$$
# 

#

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter using gradient descent. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


#

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))


# **Expected Output**:
# 
# <table style="width:100%"> 
#     <tr>
#     <td > W1 </td> 
#            <td > [[-0.59562069 -0.09991781 -2.14584584  1.82662008]
#  [-1.76569676 -0.80627147  0.51115557 -1.18258802]
#  [-1.0535704  -0.86128581  0.68284052  2.20374577]] </td> 
#   </tr> 
#   
#     <tr>
#     <td > b1 </td> 
#            <td > [[-0.04659241]
#  [-1.28888275]
#  [ 0.53405496]] </td> 
#   </tr> 
#   <tr>
#     <td > W2 </td> 
#            <td > [[-0.55569196  0.0354055   1.32964895]]</td> 
#   </tr> 
#   
#     <tr>
#     <td > b2 </td> 
#            <td > [[-0.84610769]] </td> 
#   </tr> 
# </table>
# 

# 
### 7) Conclusion
# 
# Have implemented all the functions required for building a deep neural network! 
#  
# Next, will put all these together to build two models:
# - A two-layer neural network
# - An L-layer neural network
