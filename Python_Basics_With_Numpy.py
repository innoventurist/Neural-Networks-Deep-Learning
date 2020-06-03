
# coding: utf-8

## Python Basics with Numpy 

## Set test to `"Hello World"` in the cell below to print "Hello World" 
test = "Hello World"

print ("test: " + test)

# Run the two cells

# **Expected output**:
# test: Hello World

# #1 - Building basic functions with numpy ##
# 
# Numpy is the main package for scientific computing in Python. It is maintained by a large community (www.numpy.org).
# Use several key numpy functions such as np.exp, np.log, and np.reshape. 
# 

### basic_sigmoid ###

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    s = 1/ (1 + math.exp(-x)) # function the returns the simoid as a real number x using logistic function
    
    return s


# In[4]:

print(basic_sigmoid(3))


# **Expected Output**: 
# <table style = "width:40%">
#     <tr>
#     <td>** basic_sigmoid(3) **</td> 
#         <td>0.9525741268224334 </td> 
#     </tr>
# 
# </table>

# In[20]:

x = [1, 2, 3]
basic_sigmoid(x) # will see this give an error when ran, because x is a vector.


# If $ x = (x_1, x_2, ..., x_n)$ is a row vector, then $np.exp(x)$ will apply the exponential function to every element of x. The output will thus be: $np.exp(x) = (e^{x_1}, e^{x_2}, ..., e^{x_n})$

# In[5]:

import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))


# If x is a vector, then a Python operation such as $s = x + 3$ or $s = \frac{1}{x}$ will output s as a vector of the same size as x.

# In[6]:

# example of vector operation
x = np.array([1, 2, 3])
print (x + 3)


# Info on a numpy function = [the official documentation](https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.exp.html). 
# 
# Implement the sigmoid function using numpy. 
# 
# x could now be either a real number, a vector, or a matrix.
# The data structures we use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays. 
# $$ \text{For } x \in \mathbb{R}^n \text{,     } sigmoid(x) = sigmoid\begin{pmatrix}
#     x_1  \\
#     x_2  \\
#     ...  \\
#     x_n  \\
# \end{pmatrix} = \begin{pmatrix}
#     \frac{1}{1+e^{-x_1}}  \\
#     \frac{1}{1+e^{-x_2}}  \\
#     ...  \\
#     \frac{1}{1+e^{-x_n}}  \\
# \end{pmatrix}\tag{1} $$

# In[7]:

### Sigmoid ###

import numpy as np   # this means can access numpy functions by writing np.function() instead of numpy.function()

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    s = 1/(1 + np.exp(-x))
    
    return s


x = np.array([1, 2, 3])
print(sigmoid(x))


# **Expected Output**: 
# <table>
#     <tr> 
#         <td> **sigmoid([1,2,3])**</td> 
#         <td> array([ 0.73105858,  0.88079708,  0.95257413]) </td> 
#     </tr>
# </table> 
# 

# ### Sigmoid gradient ###
# 
# Compute gradients to optimize loss functions using backpropagation. Code gradient function.
#

### sigmoid_derivative ###

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    # Compute the gradient of the sigmoid function with respect to the input x
    s = 1/(1 + np.exp(-x)) 
    ds = s * (1 - s)
    
    return ds


# In[10]:

x = np.array([1, 2, 3])
print("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))


# **Expected Output**: 
# 
# 
# <table>
#     <tr> 
#         <td> **sigmoid_derivative([1,2,3])**</td> 
#         <td> [ 0.19661193  0.10499359  0.04517666] </td> 
#     </tr>
# </table> 
# 
# 

# ### Reshaping arrays ###
# 
### image2vector ###

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    # Ger the shape of the the matrix/vector and reshape into some other dimension
    v = image.reshape(image.shape[3]*image.shape[3], image.shape[2], 1)
    
    return v


# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print("image2vector(image) = " + str(image2vector(image)))


# **Expected Output**: 
# 
# 
# <table style="width:100%">
#      <tr> 
#        <td> **image2vector(image)** </td> 
#        <td> [[ 0.67826139]
#  [ 0.29380381]
#  [ 0.90714982]
#  [ 0.52835647]
#  [ 0.4215251 ]
#  [ 0.45017551]
#  [ 0.92814219]
#  [ 0.96677647]
#  [ 0.85304703]
#  [ 0.52351845]
#  [ 0.19981397]
#  [ 0.27417313]
#  [ 0.60659855]
#  [ 0.00533165]
#  [ 0.10820313]
#  [ 0.49978937]
#  [ 0.34144279]
#  [ 0.94630077]]</td> 
#      </tr>
#     
#    
# </table>

# ### 1.4 - Normalizing rows
# 

### normalizeRows ###

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    x = x/x_norm

    return x


x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))


# **Expected Output**: 
# 
# <table style="width:60%">
# 
#      <tr> 
#        <td> **normalizeRows(x)** </td> 
#        <td> [[ 0.          0.6         0.8       ]
#  [ 0.13736056  0.82416338  0.54944226]]</td> 
#      </tr>
#     
#    
# </table>

# ### Broadcasting and the softmax function ###
# A very important concept to understand in numpy is "broadcasting".
# It is very useful for performing mathematical operations between arrays of different shapes.(http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

# Implement a softmax function using numpy. Think of softmax as a normalizing function used when the algorithm needs to classify two or more classes. 
# 
# - $ \text{for } x \in \mathbb{R}^{1\times n} \text{,     } softmax(x) = softmax(\begin{bmatrix}
#     x_1  &&
#     x_2 &&
#     ...  &&
#     x_n  
# \end{bmatrix}) = \begin{bmatrix}
#      \frac{e^{x_1}}{\sum_{j}e^{x_j}}  &&
#     \frac{e^{x_2}}{\sum_{j}e^{x_j}}  &&
#     ...  &&
#     \frac{e^{x_n}}{\sum_{j}e^{x_j}} 
# \end{bmatrix} $ 
# 
# - $\text{for a matrix } x \in \mathbb{R}^{m \times n} \text{,  $x_{ij}$ maps to the element in the $i^{th}$ row and $j^{th}$ column of $x$, thus we have: }$  $$softmax(x) = softmax\begin{bmatrix}
#     x_{11} & x_{12} & x_{13} & \dots  & x_{1n} \\
#     x_{21} & x_{22} & x_{23} & \dots  & x_{2n} \\
#     \vdots & \vdots & \vdots & \ddots & \vdots \\
#     x_{m1} & x_{m2} & x_{m3} & \dots  & x_{mn}
# \end{bmatrix} = \begin{bmatrix}
#     \frac{e^{x_{11}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{12}}}{\sum_{j}e^{x_{1j}}} & \frac{e^{x_{13}}}{\sum_{j}e^{x_{1j}}} & \dots  & \frac{e^{x_{1n}}}{\sum_{j}e^{x_{1j}}} \\
#     \frac{e^{x_{21}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{22}}}{\sum_{j}e^{x_{2j}}} & \frac{e^{x_{23}}}{\sum_{j}e^{x_{2j}}} & \dots  & \frac{e^{x_{2n}}}{\sum_{j}e^{x_{2j}}} \\

### softmax ###

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """

    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(np.exp(x), axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum
    
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))


# **Expected Output**:
# 
# <table style="width:60%">
# 
#      <tr> 
#        <td> **softmax(x)** </td> 
#        <td> [[  9.80897665e-01   8.94462891e-04   1.79657674e-02   1.21052389e-04
#     1.21052389e-04]
#  [  8.78679856e-01   1.18916387e-01   8.01252314e-04   8.01252314e-04
#     8.01252314e-04]]</td> 
#      </tr>
# </table>
# 
# If print the shapes of x_exp, x_sum and s above and rerun the assessment cell,  see that x_sum is of shape (2,1) while x_exp and s are of shape (2,5). 
# 
# **Need to remember:**
# - np.exp(x) works for any np.array x and applies the exponential function to every coordinate
# - the sigmoid function and its gradient
# - image2vector is commonly used in deep learning
# - np.reshape is widely used. In the future, you'll see that keeping your matrix/vector dimensions straight will go toward eliminating a lot of bugs. 
# - numpy has efficient built-in functions
# - broadcasting is extremely useful

# ## 2) Vectorization

import time

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# In[32]:

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# The vectorized implementation is much cleaner and more efficient. For bigger vectors/matrices, the differences in running time become even bigger. 
# 
# `np.dot()` performs a matrix-matrix or matrix-vector multiplication.

###Implement the L1 and L2 loss functions###
#
# - L1 loss is defined as:
# $$\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^m|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}$$
#

### L1 ###

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    # Implement numpy vectorized version of L1 loss using absolut value of x
    loss = abs(yhat - y).sum()

    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 " + str(L1(yhat, y)))


# **Expected Output**:
# 
# <table style="width:20%">
# 
#      <tr> 
#        <td> **L1** </td> 
#        <td> 1.1 </td> 
#      </tr>
# </table>
# 
# 
# - L2 loss is defined as:
# $$\begin{align*} & L_2(\hat{y},y) = \sum_{i=0}^m(y^{(i)} - \hat{y}^{(i)})^2 \end{align*}\tag{7}$$
#

### L2 ###

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    # Now implement L2 loss 
    loss = abs((y - yhat)**2).sum()

    
    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2  " + str(L2(yhat,y)))


# **Expected Output**: 
# <table style="width:20%">
#      <tr> 
#        <td> **L2** </td> 
#        <td> 0.43 </td> 
#      </tr>
# </table>

