import numpy as np
# ----------------------------------
def sigmoid(z):
    """
	sigmoid激活函数
	"""
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    
    return s
# ----------------------------------
def initialize_with_zeros(dim):
    """
    Argument:
    dim -- 输入数据的维度
    
    Returns:
    w -- 初始化维度为(dim, 1)的向量
    b -- 初始化标量
    """
    w = np.zeros((dim,1))
    b = 0
  
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
  
    return w, b
# ----------------------------------
def propagate(w, b, X, Y):
    """
    代价函数
  
    Argument:
    w -- 权重
    b -- 偏移量
    X -- (num_px*num_px*3, 1)维度的数据
    Y -- 维度为(1, 样本数量)标签
  
    Returns:
    cost -- 代价
    dw -- 损失相对于 w 的梯度，因此维度与 w 相同
    db -- 损失相对于 b 的梯度，因此维度与 b 相同
    """
  
    m = X.shape[1] # 样本数量
    A = sigmoid(np.dot(w.T, X) + b) # 预测值
    cost = -(1.0/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
  
    dw = (1.0/m) * np.dot(X, (A - Y).T)
    db = (1.0/m) * np.sum(A - Y)
  
    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    cost = np.squeeze(cost) # 将数组转化为向量（这里为防止bug）
    assert(cost.shape == ())
  
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
# ----------------------------------
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    w和b的迭代优化
  
    Argument:
    w -- 权重
    b -- 偏移量
    X -- (num_px*num_px*3, 1)维度的数据
    Y -- 维度为(1, 样本数量)标签
    num_iterations -- 优化迭代的次数
    learning_rate -- 学习率
    print_cost -- 如果为true，每迭代100次打印一次损失
  
    Returns:
    params -- 包含权重 w 和偏差 b 的字典
    grads -- 包含权重梯度和相对于成本函数的偏差梯度的字典
    costs -- 优化期间计算的所有成本的列表，这将用于绘制学习曲线。
    """
  
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
    
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i:%f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads
# ----------------------------------
def predict(w, b, X):
    """
    使用学习的逻辑回归参数 (w, b) 预测标签是 0 还是 1

    Arguments:
    w -- 权重
    b -- 偏移量
    X -- (num_px*num_px*3, 样本数量)维度的数据
    
    Returns:
    Y_prediction -- 一个numpy数组（向量），包含X中示例的所有预测（0/1）
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    
    assert(Y_prediction.shape == (1, m))

    return Y_prediction
# ----------------------------------
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    构建逻辑回归模型

    Arguments:
    X_train -- 训练样本 shape:(num_px * num_px * 3, m_train)
    Y_train -- 训练标签 shape:(1, m_train)
    X_test -- 测试样本 shape:(num_px * num_px * 3, m_test)
    Y_test -- 测试标签 shape:(1, m_test)
    num_iterations -- 迭代次数 默认为2000
    learning_rate -- 学习率 默认为0.5
    print_cost -- 是否打印代价
    
    Returns:
    d -- 包含模型信息的字典
    """

    w, b = initialize_with_zeros(X_train.shape[0])
    # 训练
    parameter, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # 训练结果
    w = parameter["w"]
    b = parameter["b"]
    # 预测
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # 打印预测结果
    print("训练集 预测准确率：{} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试集 预测准确率：{} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"cost": costs,
         "测试集预测正确个数": Y_prediction_test,
         "训练集预测正确个数": Y_prediction_train,
         "w": w,
         "b": b,
         "学习率": learning_rate,
         "迭代轮数": num_iterations}

    return d
  
## 最后就可以使用model函数对已经经过数据处理的训练集和测试集进行训练和预测了

