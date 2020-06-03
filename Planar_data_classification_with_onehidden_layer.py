
# coding: utf-8

### Planar data classification with one hidden layer ###
# 
# Goal:
# - Implement a 2-class classification neural network with a single hidden layer
# - Use units with a non-linear activation function, such as tanh 
# - Compute the cross entropy loss 
# - Implement forward and backward propagation
# 

# ## Packages ##
# 
# Import all the packages needed 
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import * 
import sklearn                      #provides simple and efficient tools for data mining and data analysis
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets # various various useful functions needed here

get_ipython().magic('matplotlib inline')

np.random.seed(1) # set a seed so that the results are consistent


# ## 2 - Dataset ##
# 

X, Y = load_planar_dataset() # Obtain the data set, which will load a flow 2-class data set into variables 'X' and 'Y'


# Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) and some blue (y=1) points.
# Goal is to build a model to fit this data. In other words, want the classifier to define regions as either red or blue.


# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);


# Have:
#     - a numpy-array (matrix) X that contains your features (x1, x2)
#     - a numpy-array (vector) Y that contains your labels (red:0, blue:1).
# 
# To get the shape of a numpy array: [(help)](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html)

shape_X = X.shape # Find shape of X
shape_Y = Y.shape # Find shape of Y
m = Y.shape[1]    # training set size

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


# **Expected Output**:
#        
# <table style="width:20%">
#   
#   <tr>
#     <td>**shape of X**</td>
#     <td> (2, 400) </td> 
#   </tr>
#   
#   <tr>
#     <td>**shape of Y**</td>
#     <td>(1, 400) </td> 
#   </tr>
#   
#     <tr>
#     <td>**m**</td>
#     <td> 400 </td> 
#   </tr>
#   
# </table>

### Simple Logistic Regression ###
# 
# Before building a full neural network, see how logistic regression performs on this problem. Can use sklearn's built-in functions to do that.
#
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
# Run the code to train a logistic regression classifier on the dataset.

# Can now plot the decision boundary of these models. Run the code below.

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")


# **Expected Output**:
# 
# <table style="width:20%">
#   <tr>
#     <td>**Accuracy**</td>
#     <td> 47% </td> 
#   </tr>
#   
# </table>
# 

### Neural Network model ###
# 
# Logistic regression did not work well on the "flower dataset". Now train a Neural Network with a single hidden layer.
# 
# **Here is the model**:
# <img src="images/classification_kiank.png" style="width:600px;height:300px;">
# 
# **Mathematically**:
# 
# For one example $x^{(i)}$:
# $$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1]}\tag{1}$$ 
# $$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
# $$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2]}\tag{3}$$
# $$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
# $$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$
# 
# Given the predictions on all the examples, can also compute the cost $J$ as follows: 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$
# 
# The general methodology to build a Neural Network is to:
#     1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
#     2. Initialize the model's parameters
#     3. Loop:
#         - Implement forward propagation
#         - Compute loss
#         - Implement backward propagation to get the gradients
#         - Update parameters (gradient descent)
# 
# Often build helper functions to compute steps 1-3 and then merge them into one function we call `nn_model()`.
# Once  `nn_model()` is built and learnt the right parameters, can make predictions on new data.

# ### Defining the neural network structure ####
#
### layer_sizes ###

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    # Use shapes X and Y to find n_x and n_y. Define each variable.
    n_x = X.shape[0] # size of input layer
    n_h = 4          # hardcode hidden layer size to be 4
    n_y = Y.shape[0] # size of output layer
    
    return (n_x, n_h, n_y)


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))


# **Expected Output** (these are not the sizes used for the network. They are just used to assess the function just coded).
# 
# <table style="width:20%">
#   <tr>
#     <td>**n_x**</td>
#     <td> 5 </td> 
#   </tr>
#   
#     <tr>
#     <td>**n_h**</td>
#     <td> 4 </td> 
#   </tr>
#   
#     <tr>
#     <td>**n_y**</td>
#     <td> 2 </td> 
#   </tr>
#   
# </table>

# ### Initialize the model's parameters ####
# 
#
### initialize_parameters ###

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # set up a seed so that output matches expected although the initialization is random.
    
   # Initialize weight matrices with random values
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))   # Initialize bias vectors as zeros
    
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[45]:

n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# <table style="width:90%">
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.00416758 -0.00056267]
#  [-0.02136196  0.01640271]
#  [-0.01793436 -0.00841747]
#  [ 0.00502881 -0.01245288]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ 0.]
#  [ 0.]
#  [ 0.]
#  [ 0.]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-0.01057952 -0.00909008  0.00551454  0.02292208]]</td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.]] </td> 
#   </tr>
#   
# </table>
# 
# 

# ### 4.3 - The Loop ####
# 
# Implement `forward_propagation()`.
# 
### forward_propagation ##

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
 
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)            # Use function tanh
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)            # Use function sigmoid

    
    assert(A2.shape == (1, X.shape[1]))
    # Values needed for backpropagation are stored in 'cache,' which will be given as an input to backprop
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[47]:

X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

# Note: Use the mean here just to make sure that the output matches. 
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


# **Expected Output**:
# <table style="width:50%">
#   <tr>
#     <td> 0.262818640198 0.091999045227 -1.30766601287 0.212877681719 </td> 
#   </tr>
# </table>

# Now that $A^{[2]}$ (in the Python variable "`A2`") has been computed, which contains $a^{[2](i)}$ for every example, can compute the cost function as follows:
# 
# $$J = - \frac{1}{m} \sum\limits_{i = 1}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{13}$$
# 
# Implement `compute_cost()` to compute the value of the cost $J$.
#

### compute_cost ###

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    [Note that the parameters argument is not used in this function, 
    but the auto-grader currently expects this parameter.
    Future version of this notebook will fix both the notebook 
    and the auto-grader so that `parameters` is not needed.
    For now, please include `parameters` in the function signature,
    and also when invoking this function.]
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
   
    
    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect; removes also any redundant dimensions
                                    # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float)) # can cast an array
    
    return cost


A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# **Expected Output**:
# <table style="width:20%">
#   <tr>
#     <td>**cost**</td>
#     <td> 0.693058761... </td> 
#   </tr>
#   
# </table>
#
# Implement the function `backward_propagation()`.
# 
# Backpropagation is usually the hardest (most mathematical) part in deep learning.
# Use the six equations to build a vectorized implementation:  
#
# $\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = \frac{1}{m} (a^{[2](i)} - y^{(i)})$
# 
# $\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T} $
# 
# $\frac{\partial \mathcal{J} }{ \partial b_2 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}$
# 
# $\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2}) $
# 
# $\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T $
# 
# $\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}$
# 
# - Note that $*$ denotes elementwise multiplication.
# - The notation you will use is common in deep learning coding:
#     - dW1 = $\frac{\partial \mathcal{J} }{ \partial W_1 }$
#     - db1 = $\frac{\partial \mathcal{J} }{ \partial b_1 }$
#     - dW2 = $\frac{\partial \mathcal{J} }{ \partial W_2 }$
#     - db2 = $\frac{\partial \mathcal{J} }{ \partial b_2 }$
#     

### backward_propagation ###

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']
   
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']
   
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis = 1, keepdims = True)
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


# **Expected output**:
# 
# 
# 
# <table style="width:80%">
#   <tr>
#     <td>**dW1**</td>
#     <td> [[ 0.00301023 -0.00747267]
#  [ 0.00257968 -0.00641288]
#  [-0.00156892  0.003893  ]
#  [-0.00652037  0.01618243]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**db1**</td>
#     <td>  [[ 0.00176201]
#  [ 0.00150995]
#  [-0.00091736]
#  [-0.00381422]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**dW2**</td>
#     <td> [[ 0.00078841  0.01765429 -0.00084166 -0.01022527]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**db2**</td>
#     <td> [[-0.16655712]] </td> 
#   </tr>
#   
# </table>  

# Implement the update rule. Use gradient descent. Will use (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).
# 
# General gradient descent rule: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter. 
# 

### update_parameters ###

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    
    # Update rule for each parameter using gradient descent
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters



parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# 
# <table style="width:80%">
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.00643025  0.01936718]
#  [-0.02410458  0.03978052]
#  [-0.01653973 -0.02096177]
#  [ 0.01046864 -0.05990141]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ -1.02420756e-06]
#  [  1.27373948e-05]
#  [  8.32996807e-07]
#  [ -3.20136836e-06]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-0.01041081 -0.04463285  0.01758031  0.04747113]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.00010457]] </td> 
#   </tr>
#   
# </table>  

# ### 4.4 - Integrate parts 4.1, 4.2 and 4.3 in nn_model() ####
# 
# Build neural network model in `nn_model()`.
# 
# The neural network model has to use the previous functions in the right order.


### nn_model ###

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
   
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters



X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# **Expected Output**:
# 
# <table style="width:90%">
# 
# <tr> 
#     <td> 
#         **cost after iteration 0**
#     </td>
#     <td> 
#         0.692739
#     </td>
# </tr>
# 
# <tr> 
#     <td> 
#         <center> $\vdots$ </center>
#     </td>
#     <td> 
#         <center> $\vdots$ </center>
#     </td>
# </tr>
# 
#   <tr>
#     <td>**W1**</td>
#     <td> [[-0.65848169  1.21866811]
#  [-0.76204273  1.39377573]
#  [ 0.5792005  -1.10397703]
#  [ 0.76773391 -1.41477129]]</td> 
#   </tr>
#   
#   <tr>
#     <td>**b1**</td>
#     <td> [[ 0.287592  ]
#  [ 0.3511264 ]
#  [-0.2431246 ]
#  [-0.35772805]] </td> 
#   </tr>
#   
#   <tr>
#     <td>**W2**</td>
#     <td> [[-2.45566237 -3.27042274  2.00784958  3.36773273]] </td> 
#   </tr>
#   
# 
#   <tr>
#     <td>**b2**</td>
#     <td> [[ 0.20459656]] </td> 
#   </tr>
#   
# </table>  

### Predictions ###
# 

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
   
    
    return predictions



parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))


# **Expected Output**: 
# 
# 
# <table style="width:40%">
#   <tr>
#     <td>**predictions mean**</td>
#     <td> 0.666666666667 </td> 
#   </tr>
#   
# </table>

# Run the model and see how it performs on a planar dataset.
# Run the following code to test model with a single hidden layer of $n_h$ hidden units.

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))


# **Expected Output**:
# 
# <table style="width:40%">
#   <tr>
#     <td>**Cost after iteration 9000**</td>
#     <td> 0.218607 </td> 
#   </tr>
#   
# </table>
# 

# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# **Expected Output**: 
# 
# <table style="width:15%">
#   <tr>
#     <td>**Accuracy**</td>
#     <td> 90% </td> 
#   </tr>
# </table>

# Accuracy is really high compared to Logistic Regression.
# The model has learnt the leaf patterns of the flower! Neural networks are able to learn even highly non-linear decision boundaries, unlike logistic regression. 
# 
# Now, let's try out several hidden layer sizes.

### Tuning hidden layer size  ###
# 
# Run the following code. It may take 1-2 minutes. Observation of different behaviors of the model for various hidden layer sizes.

# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

# Summary:
# - The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data. 
# - The best hidden layer size seems to be around n_h = 5. A value around here seems to  fits the data well without also incurring noticeable overfitting.
# - Will also learn later about regularization, which lets use very large models (such as n_h = 50) without much overfitting. 


### Datasets ###
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}


dataset = "noisy_moons"


X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

