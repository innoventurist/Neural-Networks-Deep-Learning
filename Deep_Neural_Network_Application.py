
# coding: utf-8

# Deep Neural Network for Image Classification: Application
# 
# Goal:
#   - See an improvement in accuracy relative to your previous logistic regression implementation.  
#   - Build and apply a deep neural network to supervised learning. 
# 

# 1) Packages

# Import all the packages needed.
# 
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *          # provides the functions implemented in previous notebooks

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)


### 2) Dataset
#
#
# Load the dataset below to get familiar with it.
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# The following code will show an image in the dataset. Can change the index and re-run the cell multiple times to see other images. 
# Example of a picture
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


#
# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# As usual, reshape and standardize the images before feeding them to the network. The code is in the cell below:
# 
# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

#

### 3) Architecture of your model

# Build a deep neural network to distinguish images.
# 
# Build two different models:
# - A 2-layer neural network
# - An L-layer deep neural network
# 
# Will then compare the performance of these models, and also try out different values for $L$. 
#
# Follow the Deep Learning methodology to build the model:
#     1. Initialize parameters / Define hyperparameters
#     2. Loop for num_iterations:
#         a. Forward propagation
#         b. Compute cost function
#         c. Backward propagation
#         d. Update parameters (using parameters, and grads from backprop) 
#     4. Use trained parameters to predict labels
# 
# Now implement those two models.

### 4) Two-layer neural network
# 
### CONSTANTS DEFINING THE MODEL ####
n_x = 12288     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

#

### two_layer_model ###

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
        
        # Compute cost
        cost = compute_cost(A2, Y)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')

        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
       
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# Run the cell below to train parameters. The cost should be decreasing. May take up to 5 minutes to run 2500 iterations.
# Check if the "Cost after iteration 0" matches the expected output below, stop the cell and try to find your error.

#

parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)


# **Expected Output**:
# <table> 
#     <tr>
#         <td> **Cost after iteration 0**</td>
#         <td> 0.6930497356599888 </td>
#     </tr>
#     <tr>
#         <td> **Cost after iteration 100**</td>
#         <td> 0.6464320953428849 </td>
#     </tr>
#     <tr>
#         <td> **...**</td>
#         <td> ... </td>
#     </tr>
#     <tr>
#         <td> **Cost after iteration 2400**</td>
#         <td> 0.048554785628770226 </td>
#     </tr>
# </table>

# 
# Now, use the trained parameters to classify images from the dataset.
# To see the predictions on the training and test sets, run the cell below.
predictions_train = predict(train_x, train_y, parameters)


# **Expected Output**:
# <table> 
#     <tr>
#         <td> **Accuracy**</td>
#         <td> 1.0 </td>
#     </tr>
# </table>

# 

predictions_test = predict(test_x, test_y, parameters)


# **Expected Output**:
# 
# <table> 
#     <tr>
#         <td> **Accuracy**</td>
#         <td> 0.72 </td>
#     </tr>
# </table>

# Note: May notice that running the model on fewer iterations (say 1500) gives better accuracy on the test set. This is called "early stopping".
# Early stopping is a way to prevent overfitting. 
# 
# It seems that the 2-layer neural network has better performance (72%) than the logistic regression implementation.
# Now, see if can do even better with an $L$-layer model.

### 5) L-layer Neural Network
# 

### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


#

### L_layer_model ###

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization.
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# Now train the model as a 4-layer neural network. 
# 
# Run the cell below to train  model. The cost should decrease on every iteration. It may take up to 5 minutes to run 2500 iterations.
# Check if the "Cost after iteration 0" matches the expected output below. If not, stop the cell and try to find your error.

# 

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)


# **Expected Output**:
# <table> 
#     <tr>
#         <td> **Cost after iteration 0**</td>
#         <td> 0.771749 </td>
#     </tr>
#     <tr>
#         <td> **Cost after iteration 100**</td>
#         <td> 0.672053 </td>
#     </tr>
#     <tr>
#         <td> **...**</td>
#         <td> ... </td>
#     </tr>
#     <tr>
#         <td> **Cost after iteration 2400**</td>
#         <td> 0.092878 </td>
#     </tr>
# </table>

# In[14]:

pred_train = predict(train_x, train_y, parameters)


# <table>
#     <tr>
#     <td>
#     **Train Accuracy**
#     </td>
#     <td>
#     0.985645933014
#     </td>
#     </tr>
# </table>

# In[15]:

pred_test = predict(test_x, test_y, parameters)


# **Expected Output**:
# 
# <table> 
#     <tr>
#         <td> **Test Accuracy**</td>
#         <td> 0.8 </td>
#     </tr>
# </table>

# It seems that your 4-layer neural network has better performance (80%) than your 2-layer neural network (72%) on the same test set. 
# 
# This is good performance for this task!
# 
# Now, learn how to obtain even higher accuracy by systematically searching for better hyperparameters (learning_rate, layers_dims, num_iterations, and many others). 

###  6) Results Analysis
# 
# Look at images the L-layer model labeled incorrectly

print_mislabeled_images(classes, test_x, test_y, pred_test)


# A few types of images the model tends to do poorly on include:
# - Cat body in an unusual position
# - Cat appears against a background of a similar color
# - Unusual cat color and species
# - Camera Angle
# - Brightness of the picture
# - Scale variation (cat is very large or small in image) 

# References:
# - for auto-reloading external module: http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
