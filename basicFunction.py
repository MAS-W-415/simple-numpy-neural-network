import numpy as np
from numpy._core.multiarray import error 

# activation functions
def relu(x):
    return np.maximum(x, 0)

def relu_d(x):
    return 1 * (x > 0)

def relu_leaky(x):
    return np.maximum(x, 0.01 * x)

def relu_leaky_d(x):
    return 1 * (x > 0) + 0.01 * (x <= 0)

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_d(x):
    return 1 - tanh(x) * tanh(x)

def noProcess(x):
    return x

# loss functions
def mse(y, y_hats):
    if type(y) is not np.ndarray:
        y = np.array([y])
    if type(y_hats) is not np.ndarray:
        y_hats = np.array([y_hats])
    if len(y) != len(y_hats):
        raise Exception("y_hats length is not equal to y length")
    return np.sum(np.power((y - y_hats), 2)) / len(y)

def mse_d(y, y_hats):
    if type(y) is not np.ndarray:
        y = np.array([y])
    if type(y_hats) is not np.ndarray:
        y_hats = np.array([y_hats])
    if len(y) != len(y_hats):
        raise Exception("y_hats length is not equal to y length")
    return 2 * (y_hats - y)

def mae(y, y_hats):
    if type(y) is not np.ndarray:
        y = np.array([y])
    if type(y_hats) is not np.ndarray:
        y_hats = np.array([y_hats])
    if len(y) != len(y_hats):
        raise Exception("y_hats length is not equal to y length")
    return np.sum(abs(y - y_hats)) / len(y)

def mae_d(y, y_hats):
    if type(y) is not np.ndarray:
        y = np.array([y])
    if type(y_hats) is not np.ndarray:
        y_hats = np.array([y_hats])
    if len(y) != len(y_hats):
        raise Exception("y_hats length is not equal to y length")
    return np.sign(y_hats - y)
    
# weight init functions
def weight_init_relu(x_size, hiddenLayerSize):
    # weight1 is n row k column (n is x length, k is hidden layer size) 
    weight1 = np.random.randn(x_size, hiddenLayerSize) * np.sqrt(2 / x_size)
    # weight2 is k + 1 row 1 column (+ 1 is for bias of hiddenlayer)
    weight2 = np.random.randn(hiddenLayerSize + 1 , 1) * np.sqrt(2 / (hiddenLayerSize + 1))
    return weight1, weight2

# loss judge functions
def raletive_rmse(y, y_hats):
    return (np.sqrt(mse(y, y_hats)) / (y.max - y.min)) < 0.05
