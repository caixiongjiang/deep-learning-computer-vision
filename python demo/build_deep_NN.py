from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward
import numpy as np


# ------------------------------------------
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- 包含网络中每一层的维度的Python List
    
    Returns:
    parameters -- Python参数字典 "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # 网络层数

    for i in range(1, L):
        parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameters["b" + str(i)] = np.zeros(layer_dims[i], 1)

        assert(parameters["W" + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert(parameters["b" + str(i)].shape == (layer_dims[i], 1))

    return parameters

# ------------------------------------------
def linear_forward(A, W, b):
    """
    实现前向传播的线性部分

    Arguments:
    A -- 来自上一层的激活结果 (或者为初始输入数据): (前一层的隐藏单元数, 样本数)
    W -- 权重矩阵: numpy array of shape (当前层的隐藏单元数, 前一层的隐藏单元数)
    b -- 偏置向量, numpy array of shape (当前层的隐藏单元数, 1)

    Returns:
    Z -- 激活函数的输入或称为预激活参数 
    cache -- Python参数字典包含"A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = (A, W, b)

    return Z, cache

# ------------------------------------------
def linear_activation_forward(A_prev, W, b, activation):
    """
    实现 线性——>激活层 的前向传播

    Arguments:
    A_prev -- 来自上层的激活结果 (或为初始输入数据): (前一层的隐藏单元数, 样本数)
    W -- 权重矩阵: numpy array of shape (当前层的隐藏单元数, 前一层的隐藏单元数)
    b -- 偏置向量, numpy array of shape (当前层的隐藏单元数, 1)
    activation -- 当前隐藏层使用的激活函数: "sigmoid" or "relu"

    Returns:
    A -- 激活函数的输出,也称为激活后值 
    cache -- Python字典包含 "线性缓存" and "激活缓存";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == (W.shape[0], A.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# ------------------------------------------
# 为了实现L层神经网络更加方便，需要将前L-1层的激活函数设置为ReLU，最后一层输出层激活函数设置为Sigmoid
def L_model_forward(X, parameters):
    """
    实现前向传播： the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- 初始数据, numpy array of shape (输入层大小, 样本数量)
    parameters -- 初始化deep网络的参数输出
    
    Returns:
    AL -- 上一层激活后的值
    caches -- cache的列表:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2
    # 前L-1层为relu激活函数
    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(i)], parameters["b" + str(i)], "relu")
        caches.append(cache)
    # 最后一层为sigmoid激活函数
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

# ------------------------------------------
def compute_cost(AL, Y):
    """
    计算代价函数 使用Logistic回归中使用的代价函数

    Arguments:
    AL -- 对应于标签的预测概率向量, shape (1, 样本数)
    Y -- 正确的样本 (for example: containing 0 if non-cat, 1 if cat), shape (1, 样本数)

    Returns:
    cost -- 交叉熵代价
    """ 

    m = Y.shape[1]
    cost = -(np.dot(np.log(AL), Y.T) + np.dot(np.log(1 - AL), (1 - Y).T)) / m

    cost = np.squeeze(cost) # 使得cost的维度是我们想要的（比如将[[17]]变成17）
    assert(cost.shape == ())

    return cost

# ------------------------------------------
def linear_backward(dZ, cache):
    """
    单层实现反向传播的线性部分(l层)

    Arguments:
    dZ -- 代价函数对于线性输出的梯度 (l层)
    cache -- 元组(A_prev, W, b) 来自当前层的前向传播

    Returns:
    dA_prev -- 代价函数对于激活的梯度(l-1层), 和A_prev相同的维度
    dW -- 代价函数对于W的梯度 (l层), 和W相同的维度
    db -- 代价函数对于b的梯度 (l层), 和b相同的维度
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m 
    dA_prev = np.dot(W.T, dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)

    return dA_prev, dW, db

# ------------------------------------------
def linear_activation_backward(dA, cache, activation):
    """
    实现 线性——>激活 过程的反向传播
    
    Arguments:
    dA -- 当前层l激活后的梯度
    cache -- 元组 (linear_cache, activation_cache) 为了有效计算后向传播而存储
    activation -- 当前层的激活函数, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- 代价函数对于激活的梯度(l-1层), 和A_prev相同的维度,和A_prev相同的维度
    dW -- 代价函数对于W的梯度 (l层), 和W相同的维度
    db -- 代价函数对于b的梯度 (l层), 和b相同的维度
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# ------------------------------------------
def L_model_backward(AL, Y, caches):
    """
    前L-1层为ReLU激活函数,最后一层为sigmoid函数的后向传播实现
    
    Arguments:
    AL -- 前向传播输出的概率向量 (L_model_forward())
    Y -- 真实值的向量 (containing 0 if non-cat, 1 if cat)
    caches -- 包含cache的列表:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- 带有渐变值的字典
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """

    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # 经过这一行转化，Y的维度和AL维度相同

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation="sigmoid")

    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(i + 2)], current_cache, activation="relu")
        grads["dA" + str(i + 1)] = dA_prev_temp
        grads["dW" + str(i + 1)] = dW_temp
        grads["db" + str(i + 1)] = db_temp

    return grads

# ------------------------------------------
def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- 更新后的参数字典
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """   

    L = len(parameters) // 2 # 神经网络中的层数

    for i in range(1, L + 1):
        parameters["W" + str(i)] -= learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]

    return parameters


