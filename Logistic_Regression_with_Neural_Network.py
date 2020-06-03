
# coding: utf-8

### Logistic Regression with a Neural Network mindset ###
# 
# Goal: to build a logistic regression classifier to recognize cats.
# This shows how to do this with a Neural Network mindset, and will also hone intuitions about deep learning.
#
### Import Packages ###
# 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy                    # used to test model with own picture at the end
from PIL import Image           # used to test model with own picture at the end
from scipy import ndimage   
from lr_utils import load_dataset

get_ipython().magic('matplotlib inline')


### Overview of the Problem set ###
# 
# Build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.
# Load the data by running the following code:

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# Added "_orig" at the end of image datasets (train and test) because going to preprocess them.
# After preprocessing, will end up with train_set_x and test_set_x (the labels train_set_y and test_set_y don't need any preprocessing).
# 
# Each line of train_set_x_orig and test_set_x_orig is an array representing an image. Can visualize an example by running the following code.
# Feel free also to change the `index` value and re-run to see other images. 

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
#
#
# Find values for m_train, m_test, num_px
m_train = train_set_x_orig.shape[0]     # number of training example
m_test = test_set_x_orig.shape[0]       # number of test examples
num_px = train_set_x_orig.shape[1]      # = height = width of a training example


print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# **Expected Output for m_train, m_test and num_px**: 
# <table style="width:15%">
#   <tr>
#     <td>**m_train**</td>
#     <td> 209 </td> 
#   </tr>
#   
#   <tr>
#     <td>**m_test**</td>
#     <td> 50 </td> 
#   </tr>
#   
#   <tr>
#     <td>**num_px**</td>
#     <td> 64 </td> 
#   </tr>
#   
# </table>
# 
#
# 

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T     # transpose to training set
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T        # transpose to test set


print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


# **Expected Output**: 
# 
# <table style="width:35%">
#   <tr>
#     <td>**train_set_x_flatten shape**</td>
#     <td> (12288, 209)</td> 
#   </tr>
#   <tr>
#     <td>**train_set_y shape**</td>
#     <td>(1, 209)</td> 
#   </tr>
#   <tr>
#     <td>**test_set_x_flatten shape**</td>
#     <td>(12288, 50)</td> 
#   </tr>
#   <tr>
#     <td>**test_set_y shape**</td>
#     <td>(1, 50)</td> 
#   </tr>
#   <tr>
#   <td>**sanity check after reshaping**</td>
#   <td>[17 31 56 22 33]</td> 
#   </tr>
# </table>
#

# Standardize the dataset.
train_set_x = train_set_x_flatten/255. # 255 = maximum value of a pixel channel
test_set_x = test_set_x_flatten/255.


### Building the parts of the algorithm ### 
# 
# Main steps for building a Neural Network:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)
# 
# Often build 1-3 separately and integrate them into one function callED `model()`.
# 
# ### Helper functions ###

# Compute using helper function sigmoid to make a prediction using np.exp

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = 1/ (1 + np.exp(-z))   
    
    return s



print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))


# **Expected Output**: 
# 
# <table>
#   <tr>
#     <td>**sigmoid([0, 2])**</td>
#     <td> [ 0.5         0.88079708]</td> 
#   </tr>
# </table>

# ### Initializing parameters ###

### initialize_with_zeros ###

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    # Implement parameter initialization
    w = np.zeros([dim, 1]) #initialize w as a vector of zeros
    b = 0
  
    assert(w.shape == (dim, 1)) 
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))


# **Expected Output**: 
# 
# 
# <table style="width:15%">
#     <tr>
#         <td>  ** w **  </td>
#         <td> [[ 0.]
#  [ 0.]] </td>
#     </tr>
#     <tr>
#         <td>  ** b **  </td>
#         <td> 0 </td>
#     </tr>
# </table>
# 
# For image inputs, w will be of shape (num_px $\times$ num_px $\times$ 3, 1).

# ### Forward and Backward propagation ###
# 
# Forward Propagation:
# - Will get X
# - Compute $A = \sigma(w^T X + b) = (a^{(1)}, a^{(2)}, ..., a^{(m-1)}, a^{(m)})$
# - Calculate the cost function: $J = -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}\log(a^{(i)})+(1-y^{(i)})\log(1-a^{(i)})$
# 
# Here are the two formulas used: 
# 
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}X(A-Y)^T\tag{7}$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (a^{(i)}-y^{(i)})\tag{8}$$


### propagate ###

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    # Implement a function `propagate()` that computes the cost function and its gradient.
    A = sigmoid(np.dot(w.T,X) + b)  # compute activation
    cost = -1/m *(np.dot(Y,np.log(A).T) + np.dot((1-Y),np.log(1 - A).T))  # compute cost
   
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1 / m *(np.dot(X,(A - Y).T))
    db = 1 / m *(np.sum(A - Y))
    

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))


# **Expected Output**:
# 
# <table style="width:50%">
#     <tr>
#         <td>  ** dw **  </td>
#       <td> [[ 0.99845601]
#      [ 2.39507239]]</td>
#     </tr>
#     <tr>
#         <td>  ** db **  </td>
#         <td> 0.00145557813678 </td>
#     </tr>
#     <tr>
#         <td>  ** cost **  </td>
#         <td> 5.801545319394553 </td>
#     </tr>
# 
# </table>

### Optimization ###
# The goal is to learn $w$ and $b$ by minimizing the cost function $J$.
# For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.
#

### optimize ###

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        # Initialize the parameters and also compute cost function and its gradient
        # Cost and gradient calculation 
        grads, cost = propagate(w,b,X,Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        w = w - learning_rate*dw
        b = b - learning_rate*db
     
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))


# **Expected Output**: 
# 
# <table style="width:40%">
#     <tr>
#        <td> **w** </td>
#        <td>[[ 0.19033591]
#  [ 0.12259159]] </td>
#     </tr>
#     
#     <tr>
#        <td> **b** </td>
#        <td> 1.92535983008 </td>
#     </tr>
#     <tr>
#        <td> **dw** </td>
#        <td> [[ 0.67752042]
#  [ 1.41625495]] </td>
#     </tr>
#     <tr>
#        <td> **db** </td>
#        <td> 0.219194504541 </td>
#     </tr>
# 
# </table>

# The previous function will output the learned w and b. Able to use w and b to predict the labels for a dataset X.
# Implement the `predict()` function. There are two steps to computing predictions:
# 
# 1. Calculate $\hat{Y} = A = \sigma(w^T X + b)$
# 
# 2. Convert the entries of a into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`.
# Can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). 

### predict ###

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X) + b)
    
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0][i] <= 0.5): 
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
        
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


# In[16]:

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))


# **Expected Output**: 
# 
# <table style="width:30%">
#     <tr>
#          <td>
#              **predictions**
#          </td>
#           <td>
#             [[ 1.  1.  0.]]
#          </td>  
#    </tr>
# 
# </table>
# 

# <font color='blue'>
# Remember
# Have implemented several functions that:
# - Initialize (w,b)
# - Optimize the loss iteratively to learn parameters (w,b):
#     - computing the cost and its gradient 
#     - updating the parameters using gradient descent
# - Use the learned (w,b) to predict the labels for a given set of examples

# ### Merge all functions into a model ###
# 
# Implement the model function. Use the following notation:
#     - Y_prediction_test for your predictions on the test set
#     - Y_prediction_train for your predictions on the train set
#     - w, costs, grads for the outputs of optimize()


### model ###

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent 
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations = 2000, learning_rate = 0.5, print_cost = False)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples 
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d


# Run the following cell to train the model.
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# **Expected Output**: 
# 
# <table style="width:40%"> 
# 
#     <tr>
#         <td> **Cost after iteration 0 **  </td> 
#         <td> 0.693147 </td>
#     </tr>
#       <tr>
#         <td> <center> $\vdots$ </center> </td> 
#         <td> <center> $\vdots$ </center> </td> 
#     </tr>  
#     <tr>
#         <td> **Train Accuracy**  </td> 
#         <td> 99.04306220095694 % </td>
#     </tr>
# 
#     <tr>
#         <td>**Test Accuracy** </td> 
#         <td> 70.0 % </td>
#     </tr>
# </table> 
# 
# 
# 

# Note: Training accuracy is close to 100%. Good sanity check: model is working and has high enough capacity to fit the training data.
# Test accuracy is 68%--not bad for this simple model, given the small dataset we used and that logistic regression is a linear classifier.
# 
# Also, the model is clearly overfitting the training data.  Could fix this using by using regularization.
# Using the code below (and changing the `index` variable), can look at predictions on pictures of the test set.

# Example of a picture that was wrongly classified.
index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")


# Plot the cost function and the gradients.
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# Note:
# can see the cost decreasing = the parameters are being learned. However, could train the model even more on the training set.
# Try to increase the number of iterations in the cell above and rerun the cells. Will see the training set accuracy goes up, but the test set accuracy goes down (overfitting). 

# ### Further analysis (optional/ungraded exercise) ##
# 
# Have built first image classification model. Now, let's analyze it further, and examine possible choices for the learning rate $\alpha$. 

# #### Choice of learning rate ####
# 
# Note:
# In order for Gradient Descent to work, must choose the learning rate wisely. The learning rate $\alpha$ determines how rapidly the parameters are updated.
# If the learning rate is too large, may "overshoot" the optimal value. If it is too small we will need too many iterations to converge to the best values.
# It's crucial to use a well-tuned learning rate.
# 
# Compare the learning curve of the model with several choices of learning rates. Run the cell below. 

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# Note: 
# - Different learning rates give different costs and thus different predictions results.
# - If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (for this, using 0.01 still eventually ends up at a good value for the cost). 
# - A lower cost doesn't mean a better model. Have to check if there is possibly overfitting. It happens when the training accuracy is higher than the test accuracy.
# - In deep learning, it's recommended to: 
#     - Choose the learning rate that better minimizes the cost function.
#     - If the model overfits, use other techniques to reduce overfitting.
#
#
# Summary:
# 1. Preprocessing the dataset is important.
# 2. Implemented each function separately: initialize(), propagate(), optimize(). Then you built a model().
# 3. Tuning the learning rate (which is an example of a "hyperparameter") can make a big difference to the algorithm. 


# Bibliography:
# - http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# - https://stats.stackexchange.com/questions/211436/why-do-we-normalize-images-by-subtracting-the-datasets-image-mean-and-not-the-c
