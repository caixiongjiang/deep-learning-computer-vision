import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
def sigmoid(z):
    """
	sigmoid激活函数
	"""
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    
    return s
# -----------------------------------------
def layer_sizes(X, Y):
    """
    Arguments:
    X -- 输入数据 (输入层大小, 样本数量)
    Y -- 标签 (输出层大小, 样本数量)
    
    Returns:
    n_x -- 输入层的大小
    n_h -- 隐藏层的大小
    n_y -- 输出层的大小
    """

    n_x = X.shape[0] # 输入层的大小
    n_h = 4
    n_y = Y.shape[0] # 输出层的大小

    return (n_x, n_h, n_y)


# -----------------------------------------
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- 输入层的大小
    n_h -- 隐藏层的大小
    n_y -- 输出层的大小

    Returns:
    params -- 初始化参数的字典:
              W1 -- weight matrix of shape (n_h, n_x)
              b1 -- bias vector of shape (n_h, 1)
              W2 -- weight matrix of shape (n_y, n_h)
              b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2) # 设置随机种子

    W1 = np.random.randn((n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand((n_y, n_h))
    b2 = np.zeros((n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# -----------------------------------------
def forward_propagation(X, parameters):
    """
    前向传播计算

    Argument:
    X -- 输入数据 (n_x, m)
    parameters -- 初始化参数的字典 (output of initialization function)
    
    Returns:
    A2 -- 第二层sigmoid激活函数输出的结果
    cache -- 中间权向量的字典 "Z1", "A1", "Z2" and "A2"
    """    

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1 # b1使用广播技术自动扩充
    A1 = np.tanh(Z1) # 隐藏层使用tanh激活函数
    Z2 = np.dot(W2, A1) + b2 # b2使用广播技术自动扩充
    A2 = sigmoid(Z2) # 输出层使用sigmoid激活函数

    assert(A2.shape == (1, X.shape[1])) # X.shape[1]代表样本数量

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

# -----------------------------------------
def compute_cost(A2, Y, parameters):
    """
    计算代价函数 (13)
    
    Arguments:
    A2 -- 第二层sigmoid激活函数输出的结果 维度(1, number of examples)
    Y -- 正确的标签 维度(1, number of examples)
    parameters -- 初始化参数的字典 W1, b1, W2 and b2
    
    Returns:
    cost -- 代价函数结果
    """

    m = Y.shape[1] # 样本数量

    # 计算代价函数
    # np.multiply(X, Y)是指X和Y对应位置两两相乘
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -np.sum(logprobs) / m

    cost = np.squeeze(cost) # 确保代价为我们期望的维度
    
    # isinstance() 函数来判断一个对象是否是一个已知的类型
    assert(isinstance(cost, float))

    return cost

# -----------------------------------------
def backward_propagation(parameters, cache, X, Y):
    """
    后向传播
    
    Arguments:
    parameters -- 参数初始化字典 
    cache -- 中间权向量的字典 "Z1", "A1", "Z2" and "A2"
    X -- 输入数据 维度(2, number of examples)
    Y -- 正确标签 维度(1, number of examples)
    
    Returns:
    grads -- 参数渐变的字典
    """

    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    # 后向传播计算
    # tanh()函数的导数为 g'(a) = 1 - a^2
    
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2)) # 1-np.power(A1, 2)为tanh的导数
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/ m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

# -----------------------------------------
def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    中间权重向量更新
    
    Arguments:
    parameters -- 更新前的参数 
    grads -- 用于参数更新的逆向传播参数
    
    Returns:
    parameters -- 更新后的参数
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # 权重向量更新
    W1 = W1 - dW1 * learning_rate
    b1 = b1 - db1 * learning_rate
    W2 = W2 - dW2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

# -----------------------------------------
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- 循环迭代的次数
    print_cost -- 如果为True,每1000次打印一次代价
    
    Returns:
    parameters -- 训练好的参数，用于预测
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # 循环
    for i in range(0, num_iterations):
        # 计算前向传播
        A2, cache = forward_propagation(X, parameters)
        
        # 计算代价
        cost = compute_cost(X, parameters)

        # 计算后向传播
        grads = backward_propagation(parameters, cache, X, Y)

        # 更新权向量
        parameters = update_parameters(parameters, grads) # 学习率直接设置为默认值1.2

        if print_cost and (i % 1000):
            print("第i次迭代之后的代价为:" + str(cost))

    return parameters


# ----------------------------------------- 
def predict(parameters, X):
    """
    通过训练好的权重来预测X的类型
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- 输入数据 维度 (n_x, m)
    
    Returns
    predictions -- 模型预测的结果 (red: 0 / blue: 1)
    """

    A2, cache = forward_propagation(X, parameters)
    prediction = (A2 > 0.5) # sigmoid函数的判别方式

    return prediction

# -----------------------------------------
# 使用：

# 构建双层神经网络模型
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# 绘制决策边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y[0, :])
plt.title("Decision Boundary for hidden layer size " + str(4))