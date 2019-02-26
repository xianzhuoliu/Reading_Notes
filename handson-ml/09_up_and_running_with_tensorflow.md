
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#tensorflow的优势" data-toc-modified-id="tensorflow的优势-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>tensorflow的优势</a></span></li><li><span><a href="#Creating-and-running-a-graph-建立第一张Graph在Session中运行" data-toc-modified-id="Creating-and-running-a-graph-建立第一张Graph在Session中运行-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Creating and running a graph 建立第一张Graph在Session中运行</a></span><ul class="toc-item"><li><span><a href="#建立图-(Construction-Phase)" data-toc-modified-id="建立图-(Construction-Phase)-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>建立图 (Construction Phase)</a></span></li><li><span><a href="#在sess中initialize和run-(Excution-Phase)" data-toc-modified-id="在sess中initialize和run-(Excution-Phase)-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>在sess中initialize和run (Excution Phase)</a></span></li><li><span><a href="#简易写法with-tf.Session()-as-sess:" data-toc-modified-id="简易写法with-tf.Session()-as-sess:-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>简易写法with tf.Session() as sess:</a></span></li><li><span><a href="#global_variables_initializer()" data-toc-modified-id="global_variables_initializer()-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>global_variables_initializer()</a></span></li><li><span><a href="#自己设自己为默认session:-InteractiveSession()" data-toc-modified-id="自己设自己为默认session:-InteractiveSession()-2.5"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>自己设自己为默认session: InteractiveSession()</a></span></li></ul></li><li><span><a href="#Managing-graphs-管理图" data-toc-modified-id="Managing-graphs-管理图-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Managing graphs 管理图</a></span><ul class="toc-item"><li><span><a href="#创建的任何节点都会加入到默认的graph中" data-toc-modified-id="创建的任何节点都会加入到默认的graph中-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>创建的任何节点都会加入到默认的graph中</a></span></li><li><span><a href="#可以管理过个独立的graph" data-toc-modified-id="可以管理过个独立的graph-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>可以管理过个独立的graph</a></span></li><li><span><a href="#tf.reset_default_graph()或者重启kernel(shell)" data-toc-modified-id="tf.reset_default_graph()或者重启kernel(shell)-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>tf.reset_default_graph()或者重启kernel(shell)</a></span></li></ul></li><li><span><a href="#node和variable的生命周期" data-toc-modified-id="node和variable的生命周期-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>node和variable的生命周期</a></span><ul class="toc-item"><li><span><a href="#每次sess.run(也叫graph-run）的时候，都把图重新跑一遍，即使他们共享一些值" data-toc-modified-id="每次sess.run(也叫graph-run）的时候，都把图重新跑一遍，即使他们共享一些值-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>每次sess.run(也叫graph run）的时候，都把图重新跑一遍，即使他们共享一些值</a></span></li><li><span><a href="#概括" data-toc-modified-id="概括-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>概括</a></span></li></ul></li><li><span><a href="#Linear-Regression-线性回归" data-toc-modified-id="Linear-Regression-线性回归-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Linear Regression 线性回归</a></span><ul class="toc-item"><li><span><a href="#tensor是什么" data-toc-modified-id="tensor是什么-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>tensor是什么</a></span></li><li><span><a href="#Using-the-Normal-Equation" data-toc-modified-id="Using-the-Normal-Equation-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Using the Normal Equation</a></span></li><li><span><a href="#Using-Batch-Gradient-Descent-批梯度下降" data-toc-modified-id="Using-Batch-Gradient-Descent-批梯度下降-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Using Batch Gradient Descent 批梯度下降</a></span><ul class="toc-item"><li><span><a href="#先归一化" data-toc-modified-id="先归一化-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>先归一化</a></span></li><li><span><a href="#手动计算梯度-Manually-computing-the-gradients" data-toc-modified-id="手动计算梯度-Manually-computing-the-gradients-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>手动计算梯度 Manually computing the gradients</a></span></li><li><span><a href="#自动计算梯度-Using-autodiff" data-toc-modified-id="自动计算梯度-Using-autodiff-5.3.3"><span class="toc-item-num">5.3.3&nbsp;&nbsp;</span>自动计算梯度 Using autodiff</a></span></li><li><span><a href="#Using-a-GradientDescentOptimizer-使用Optimizer" data-toc-modified-id="Using-a-GradientDescentOptimizer-使用Optimizer-5.3.4"><span class="toc-item-num">5.3.4&nbsp;&nbsp;</span>Using a GradientDescentOptimizer 使用Optimizer</a></span></li><li><span><a href="#Using-a-momentum-optimizer" data-toc-modified-id="Using-a-momentum-optimizer-5.3.5"><span class="toc-item-num">5.3.5&nbsp;&nbsp;</span>Using a momentum optimizer</a></span></li></ul></li></ul></li><li><span><a href="#Feeding-data-to-the-training-algorithm" data-toc-modified-id="Feeding-data-to-the-training-algorithm-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Feeding data to the training algorithm</a></span><ul class="toc-item"><li><span><a href="#Placeholder-nodes" data-toc-modified-id="Placeholder-nodes-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Placeholder nodes</a></span></li><li><span><a href="#Mini-batch-Gradient-Descent-实施小批梯度下降" data-toc-modified-id="Mini-batch-Gradient-Descent-实施小批梯度下降-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Mini-batch Gradient Descent 实施小批梯度下降</a></span></li></ul></li><li><span><a href="#Saving-and-restoring-a-model-保存和恢复一个模型" data-toc-modified-id="Saving-and-restoring-a-model-保存和恢复一个模型-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Saving and restoring a model 保存和恢复一个模型</a></span><ul class="toc-item"><li><span><a href="#恢复模型中的variable" data-toc-modified-id="恢复模型中的variable-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>恢复模型中的variable</a></span></li><li><span><a href="#恢复图结构-graph-strucutre" data-toc-modified-id="恢复图结构-graph-strucutre-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>恢复图结构 graph strucutre</a></span></li></ul></li><li><span><a href="#Visualizing-the-graph" data-toc-modified-id="Visualizing-the-graph-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Visualizing the graph</a></span><ul class="toc-item"><li><span><a href="#inside-Jupyter" data-toc-modified-id="inside-Jupyter-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>inside Jupyter</a></span></li><li><span><a href="#Using-TensorBoard" data-toc-modified-id="Using-TensorBoard-8.2"><span class="toc-item-num">8.2&nbsp;&nbsp;</span>Using TensorBoard</a></span></li></ul></li><li><span><a href="#Name-scopes" data-toc-modified-id="Name-scopes-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Name scopes</a></span></li><li><span><a href="#Modularity-模块化" data-toc-modified-id="Modularity-模块化-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Modularity 模块化</a></span></li><li><span><a href="#Sharing-Variables-共用变量" data-toc-modified-id="Sharing-Variables-共用变量-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Sharing Variables 共用变量</a></span><ul class="toc-item"><li><span><a href="#像parameter一样传递进函数" data-toc-modified-id="像parameter一样传递进函数-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>像parameter一样传递进函数</a></span></li><li><span><a href="#把变量绑定为函数的一个属性" data-toc-modified-id="把变量绑定为函数的一个属性-11.2"><span class="toc-item-num">11.2&nbsp;&nbsp;</span>把变量绑定为函数的一个属性</a></span></li><li><span><a href="#使用tf.get_variable()方法创建和复用变量variable" data-toc-modified-id="使用tf.get_variable()方法创建和复用变量variable-11.3"><span class="toc-item-num">11.3&nbsp;&nbsp;</span>使用tf.get_variable()方法创建和复用变量variable</a></span><ul class="toc-item"><li><span><a href="#指定复用" data-toc-modified-id="指定复用-11.3.1"><span class="toc-item-num">11.3.1&nbsp;&nbsp;</span>指定复用</a></span></li></ul></li></ul></li><li><span><a href="#Extra-material" data-toc-modified-id="Extra-material-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Extra material</a></span><ul class="toc-item"><li><span><a href="#name_scope示例" data-toc-modified-id="name_scope示例-12.1"><span class="toc-item-num">12.1&nbsp;&nbsp;</span>name_scope示例</a></span></li><li><span><a href="#Strings-字符串" data-toc-modified-id="Strings-字符串-12.2"><span class="toc-item-num">12.2&nbsp;&nbsp;</span>Strings 字符串</a></span></li><li><span><a href="#Autodiff" data-toc-modified-id="Autodiff-12.3"><span class="toc-item-num">12.3&nbsp;&nbsp;</span>Autodiff</a></span></li></ul></li><li><span><a href="#Exercise-solutions-一个完整的tensorflow逻辑回归流程" data-toc-modified-id="Exercise-solutions-一个完整的tensorflow逻辑回归流程-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Exercise solutions 一个完整的tensorflow逻辑回归流程</a></span><ul class="toc-item"><li><span><a href="#1.-to-11." data-toc-modified-id="1.-to-11.-13.1"><span class="toc-item-num">13.1&nbsp;&nbsp;</span>1. to 11.</a></span></li><li><span><a href="#12.-Logistic-Regression-with-Mini-Batch-Gradient-Descent-using-TensorFlow" data-toc-modified-id="12.-Logistic-Regression-with-Mini-Batch-Gradient-Descent-using-TensorFlow-13.2"><span class="toc-item-num">13.2&nbsp;&nbsp;</span>12. Logistic Regression with Mini-Batch Gradient Descent using TensorFlow</a></span></li></ul></li></ul></div>

**Chapter 9 – Up and running with TensorFlow**

# tensorflow的优势
- python API: tensorflow.contrib.learn
- 简化的python API: tensorflow.contrib.slim
- 几乎你能想到的任何神经网络结构
- C++ API
- 谷歌云服务
- 社区：github/jtoy/awesome-tensorflow

# Creating and running a graph 建立第一张Graph在Session中运行
## 建立图 (Construction Phase)


```python
import tensorflow as tf

reset_graph()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2
```


```python
f
```




    <tf.Tensor 'add_1:0' shape=() dtype=int32>



## 在sess中initialize和run (Excution Phase)


```python
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
```

    42
    


```python
sess.close()
```

## 简易写法with tf.Session() as sess:
把sess设定为默认的session,而且会自动关闭session


```python
with tf.Session() as sess:
    x.initializer.run() # 等价于tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval()  # 等价于tf.get_default_session().run(f)
```


```python
result
```




    42



## global_variables_initializer()
这个不是马上初始化所有变量Variable，而是创建一个节点当run这个节点的时候进行初始化


```python
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval() # 输出tensor f！
```


```python
result
```




    42




```python
init = tf.global_variables_initializer()
```

## 自己设自己为默认session: InteractiveSession()


```python
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
```

    42
    


```python
sess.close()
```


```python
result
```




    42



# Managing graphs 管理图
## 创建的任何节点都会加入到默认的graph中


```python
reset_graph()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()
```




    True



## 可以管理过个独立的graph


```python
graph = tf.Graph()
with graph.as_default(): # 作为临时的default graph
    x2 = tf.Variable(2)

x2.graph is graph
```




    True




```python
x2.graph is tf.get_default_graph() # False 出了with就不是默认的graph了
```




    False



## tf.reset_default_graph()或者重启kernel(shell)
在jupyter或者IDE中，有时会重复地在默认graph中添加node，我们要在每次执行前reset默认graph（清空）

# node和variable的生命周期
## 每次sess.run(也叫graph run）的时候，都把图重新跑一遍，即使他们共享一些值
求y和z的时候都会从头把x和w求一遍，先确定y和z的依赖


```python
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15
```

    10
    15
    

如果要更高效地计算y和z，也就是在不同的graph run中分享共同依赖的节点


```python
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15
```

    10
    15
    

## 概括
- node的生命周期只有一次graph run
- variable的什么周期在初始化时开始，session结束时结束（如果再要用需要重新初始化）

# Linear Regression 线性回归

## tensor是什么
输入和输出都是多维array，所以叫tensor

## Using the Normal Equation


```python
import numpy as np
from sklearn.datasets import fetch_california_housing

reset_graph()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")  #本来target只是个一维的向量，要让它变为2维的，才能相乘。
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
```


```python
theta_value
```




    array([[-3.7185181e+01],
           [ 4.3633747e-01],
           [ 9.3952334e-03],
           [-1.0711310e-01],
           [ 6.4479220e-01],
           [-4.0338000e-06],
           [-3.7813708e-03],
           [-4.2348403e-01],
           [-4.3721911e-01]], dtype=float32)



Compare with pure NumPy


```python
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)
```

    [[-3.69419202e+01]
     [ 4.36693293e-01]
     [ 9.43577803e-03]
     [-1.07322041e-01]
     [ 6.45065694e-01]
     [-3.97638942e-06]
     [-3.78654265e-03]
     [-4.21314378e-01]
     [-4.34513755e-01]]
    

Compare with Scikit-Learn


```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])
```

    [[-3.69419202e+01]
     [ 4.36693293e-01]
     [ 9.43577803e-03]
     [-1.07322041e-01]
     [ 6.45065694e-01]
     [-3.97638942e-06]
     [-3.78654265e-03]
     [-4.21314378e-01]
     [-4.34513755e-01]]
    

## Using Batch Gradient Descent 批梯度下降
### 先归一化

推荐使用sklearn的preprocessing

Gradient Descent requires scaling the feature vectors first. We could do this using TF, but let's just use Scikit-Learn for now.


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
```


```python
print(scaled_housing_data_plus_bias.mean(axis=0))
print(scaled_housing_data_plus_bias.mean(axis=1))
print(scaled_housing_data_plus_bias.mean())
print(scaled_housing_data_plus_bias.shape)
```

    [ 1.00000000e+00  6.60969987e-17  5.50808322e-18  6.60969987e-17
     -1.06030602e-16 -1.10161664e-17  3.44255201e-18 -1.07958431e-15
     -8.52651283e-15]
    [ 0.38915536  0.36424355  0.5116157  ... -0.06612179 -0.06360587
      0.01359031]
    0.11111111111111005
    (20640, 9)
    

### 手动计算梯度 Manually computing the gradients


```python
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
```

    Epoch 0 MSE = 9.161542
    Epoch 100 MSE = 0.7145004
    Epoch 200 MSE = 0.56670487
    Epoch 300 MSE = 0.5555718
    Epoch 400 MSE = 0.5488112
    Epoch 500 MSE = 0.5436363
    Epoch 600 MSE = 0.5396291
    Epoch 700 MSE = 0.5365092
    Epoch 800 MSE = 0.53406775
    Epoch 900 MSE = 0.5321473
    


```python
best_theta
```




    array([[ 2.0685523 ],
           [ 0.8874027 ],
           [ 0.14401656],
           [-0.34770882],
           [ 0.36178368],
           [ 0.00393811],
           [-0.04269556],
           [-0.6614529 ],
           [-0.6375279 ]], dtype=float32)



### 自动计算梯度 Using autodiff

使用tf.gradients(mse, [theta])[0]方法，计算关于[theta]中每一个变量的梯度，返回一个关于每一变量梯度的列表

Same as above except for the `gradients = ...` line:


```python
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
```


```python
gradients = tf.gradients(mse, [theta])[0]
```

举一个数学计算的例子，分别对两个变量进行求导

How could you find the partial derivatives of the following function with regards to `a` and `b`?


```python
def my_func(a, b):
    z = 0
    for i in range(100):
        z = a * np.cos(z + i) + z * np.sin(b - i)
    return z
```


```python
reset_graph()

a = tf.Variable(0.2, name="a")
b = tf.Variable(0.3, name="b")
z = tf.constant(0.0, name="z0")
for i in range(100):
    z = a * tf.cos(z + i) + z * tf.sin(b - i)

grads = tf.gradients(z, [a, b])
init = tf.global_variables_initializer()
```

Let's compute the function at $a=0.2$ and $b=0.3$, and the partial derivatives at that point with regards to $a$ and with regards to $b$:


```python
with tf.Session() as sess:
    init.run()
    print(z.eval())
    print(sess.run(grads))
```

    -0.21253741
    [-1.1388495, 0.19671397]
    

### Using a GradientDescentOptimizer 使用Optimizer
可以不用考虑求梯度再去最小化MSE，可以直接使用optimizer优化器，开箱即用

把gradients和trainiing_op这两句替换成以下两句即可


```python
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
```


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```


```python
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
```

    Epoch 0 MSE = 9.161542
    Epoch 100 MSE = 0.7145004
    Epoch 200 MSE = 0.56670487
    Epoch 300 MSE = 0.5555718
    Epoch 400 MSE = 0.54881126
    Epoch 500 MSE = 0.5436363
    Epoch 600 MSE = 0.53962916
    Epoch 700 MSE = 0.5365092
    Epoch 800 MSE = 0.53406775
    Epoch 900 MSE = 0.5321473
    Best theta:
    [[ 2.0685523 ]
     [ 0.8874027 ]
     [ 0.14401656]
     [-0.3477088 ]
     [ 0.36178365]
     [ 0.00393811]
     [-0.04269556]
     [-0.66145283]
     [-0.6375278 ]]
    

### Using a momentum optimizer


```python
reset_graph()

n_epochs = 1000
learning_rate = 0.01

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
```


```python
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9)
```


```python
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
```


```python
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        sess.run(training_op)
    
    best_theta = theta.eval()

print("Best theta:")
print(best_theta)
```

    Best theta:
    [[ 2.068558  ]
     [ 0.82962847]
     [ 0.11875335]
     [-0.26554456]
     [ 0.3057109 ]
     [-0.00450249]
     [-0.03932662]
     [-0.8998645 ]
     [-0.8705207 ]]
    

# Feeding data to the training algorithm

## Placeholder nodes
如果我们要实施小批梯度下降，每一次迭代都换X和Y，最方便的方法就是使用placeholder节点

这些节点不进行任何计算，只是在run的时候把你feed的数据输出。如果没有feed数据给它会报错

如果指定给某一个维度None，这意味着可以任何大小，如下面的A。但是喂给A的数据必须是二维的，不管是多少行，需要是二维的就行

如果是shape=()，则说明是个标量，如0.0


```python
reset_graph()

A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})

print(B_val_1)
```

    [[6. 7. 8.]]
    


```python
print(B_val_2)
```

    [[ 9. 10. 11.]
     [12. 13. 14.]]
    

## Mini-batch Gradient Descent 实施小批梯度下降


```python
n_epochs = 1000
learning_rate = 0.01
```


```python
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
```


```python
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
```


```python
n_epochs = 10
```


```python
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
```


```python
def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
```


```python
best_theta
```




    array([[ 2.0703337 ],
           [ 0.8637145 ],
           [ 0.12255152],
           [-0.31211877],
           [ 0.38510376],
           [ 0.00434168],
           [-0.0123295 ],
           [-0.83376896],
           [-0.8030471 ]], dtype=float32)



# Saving and restoring a model 保存和恢复一个模型
可能想每隔一段时间保存checkpoints，当电脑崩了的时候还能恢复最后一个checkpoint

怎么保存呢？**只需要在construction phase（在所有variable node所有节点都创建完毕，图构建完毕的时候）后面**创建一个Saver node，在execution phase里，当你想保存的，只需要call这个saver节点，并传session和checkpoint的地址给它。


```python
reset_graph()

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
error = y_pred - y                                                                    # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())                                # not shown
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)
    
    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")
```

    Epoch 0 MSE = 9.161542
    Epoch 100 MSE = 0.7145004
    Epoch 200 MSE = 0.56670487
    Epoch 300 MSE = 0.5555718
    Epoch 400 MSE = 0.54881126
    Epoch 500 MSE = 0.5436363
    Epoch 600 MSE = 0.53962916
    Epoch 700 MSE = 0.5365092
    Epoch 800 MSE = 0.53406775
    Epoch 900 MSE = 0.5321473
    


```python
best_theta
```




    array([[ 2.0685523 ],
           [ 0.8874027 ],
           [ 0.14401656],
           [-0.3477088 ],
           [ 0.36178365],
           [ 0.00393811],
           [-0.04269556],
           [-0.66145283],
           [-0.6375278 ]], dtype=float32)



## 恢复模型中的variable
在execution phase的最开始，**不要使用初始化节点（不要使用init节点）**，call resotore方法！


```python
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval() # eval只是输出tenosr的方法
```

    INFO:tensorflow:Restoring parameters from /tmp/my_model_final.ckpt
    

比较模型加载的theta和计算theta是否完全相等


```python
np.allclose(best_theta, best_theta_restored)
```




    True



可以指定想恢复或保存的参数，不用恢复或保存所有参数，比如想恢复某一name（在tf.Variable都会传一个name进去嘛）下面的参数

If you want to have a saver that loads and restores `theta` with a different name, such as `"weights"`:


```python
saver = tf.train.Saver({"weights": theta})
```

## 恢复图结构 graph strucutre
saver节点在保存变量的时候还保存了图的结构，以.meta结尾的文件。

要方便地恢复图的结构，只需要在建立图的阶段construction phase使用函数`tf.train.import_meta_graph()`即可

同时还能获取图中的variable的值，`tf.get_default_graph().get_tensor_by_name("theta:0")`在刚刚恢复的图中，获取tensor的值（这一块自己还没有试过，不是很懂）

By default the saver also saves the graph structure itself in a second file with the extension `.meta`. You can use the function `tf.train.import_meta_graph()` to restore the graph structure. This function loads the graph into the default graph and returns a `Saver` that can then be used to restore the graph state (i.e., the variable values):


```python
reset_graph()
# notice that we start with an empty graph.

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book
```

    INFO:tensorflow:Restoring parameters from /tmp/my_model_final.ckpt
    


```python
np.allclose(best_theta, best_theta_restored)
```




    True



This means that you can import a pretrained model without having to have the corresponding Python code to build the graph. This is very handy when you keep tweaking and saving your model: you can load a previously saved model without having to search for the version of the code that built it.

# Visualizing the graph
## inside Jupyter

没有试过，不是很懂

To visualize the graph within Jupyter, we will use a TensorBoard server available online at https://tensorboard.appspot.com/ (so this will not work if you do not have Internet access).  As far as I can tell, this code was originally written by Alex Mordvintsev in his [DeepDream tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb). Alternatively, you could use a tool like [tfgraphviz](https://github.com/akimach/tfgraphviz).


```python
from tensorflow_graph_in_jupyter import show_graph
```


```python
show_graph(tf.get_default_graph())
```



        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="
        <script src=&quot;//cdnjs.cloudflare.com/ajax/libs/polymer/0.3.3/platform.js&quot;></script>
        <script>
          function load() {
            document.getElementById(&quot;graph0.3745401188473625&quot;).pbtxt = 'node {\n  name: &quot;save/RestoreV2/shape_and_slices&quot;\n  op: &quot;Const&quot;\n  device: &quot;/device:CPU:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_STRING\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_STRING\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        string_val: &quot;&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;save/RestoreV2/tensor_names&quot;\n  op: &quot;Const&quot;\n  device: &quot;/device:CPU:0&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_STRING\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_STRING\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        string_val: &quot;theta&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;save/SaveV2/shape_and_slices&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_STRING\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_STRING\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        string_val: &quot;&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;save/SaveV2/tensor_names&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_STRING\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_STRING\n        tensor_shape {\n          dim {\n            size: 1\n          }\n        }\n        string_val: &quot;theta&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;save/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_STRING\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_STRING\n        tensor_shape {\n        }\n        string_val: &quot;model&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;save/RestoreV2&quot;\n  op: &quot;RestoreV2&quot;\n  input: &quot;save/Const&quot;\n  input: &quot;save/RestoreV2/tensor_names&quot;\n  input: &quot;save/RestoreV2/shape_and_slices&quot;\n  device: &quot;/device:CPU:0&quot;\n  attr {\n    key: &quot;dtypes&quot;\n    value {\n      list {\n        type: DT_FLOAT\n      }\n    }\n  }\n}\nnode {\n  name: &quot;GradientDescent/learning_rate&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 0.009999999776482582\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/mse_grad/Const_1&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 20640.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/mse_grad/Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\240P\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/mse_grad/Reshape/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\001\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/grad_ys_0&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/Shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n          }\n        }\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/Fill&quot;\n  op: &quot;Fill&quot;\n  input: &quot;gradients/Shape&quot;\n  input: &quot;gradients/grad_ys_0&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;index_type&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;gradients/mse_grad/Reshape&quot;\n  op: &quot;Reshape&quot;\n  input: &quot;gradients/Fill&quot;\n  input: &quot;gradients/mse_grad/Reshape/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tshape&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;gradients/mse_grad/Tile&quot;\n  op: &quot;Tile&quot;\n  input: &quot;gradients/mse_grad/Reshape&quot;\n  input: &quot;gradients/mse_grad/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tmultiples&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n}\nnode {\n  name: &quot;gradients/mse_grad/truediv&quot;\n  op: &quot;RealDiv&quot;\n  input: &quot;gradients/mse_grad/Tile&quot;\n  input: &quot;gradients/mse_grad/Const_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;gradients/Square_grad/Const&quot;\n  op: &quot;Const&quot;\n  input: &quot;^gradients/mse_grad/truediv&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 2.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Const&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\000\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;theta&quot;\n  op: &quot;VariableV2&quot;\n  attr {\n    key: &quot;container&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;shape&quot;\n    value {\n      shape {\n        dim {\n          size: 9\n        }\n        dim {\n          size: 1\n        }\n      }\n    }\n  }\n  attr {\n    key: &quot;shared_name&quot;\n    value {\n      s: &quot;&quot;\n    }\n  }\n}\nnode {\n  name: &quot;save/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;theta&quot;\n  input: &quot;save/RestoreV2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@theta&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;save/restore_all&quot;\n  op: &quot;NoOp&quot;\n  input: &quot;^save/Assign&quot;\n}\nnode {\n  name: &quot;save/SaveV2&quot;\n  op: &quot;SaveV2&quot;\n  input: &quot;save/Const&quot;\n  input: &quot;save/SaveV2/tensor_names&quot;\n  input: &quot;save/SaveV2/shape_and_slices&quot;\n  input: &quot;theta&quot;\n  attr {\n    key: &quot;dtypes&quot;\n    value {\n      list {\n        type: DT_FLOAT\n      }\n    }\n  }\n}\nnode {\n  name: &quot;save/control_dependency&quot;\n  op: &quot;Identity&quot;\n  input: &quot;save/Const&quot;\n  input: &quot;^save/SaveV2&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_STRING\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@save/Const&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;theta/read&quot;\n  op: &quot;Identity&quot;\n  input: &quot;theta&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@theta&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform/max&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: 1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform/min&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n        }\n        float_val: -1.0\n      }\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform/sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;random_uniform/max&quot;\n  input: &quot;random_uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform/shape&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_INT32\n        tensor_shape {\n          dim {\n            size: 2\n          }\n        }\n        tensor_content: &quot;\\t\\000\\000\\000\\001\\000\\000\\000&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform/RandomUniform&quot;\n  op: &quot;RandomUniform&quot;\n  input: &quot;random_uniform/shape&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;seed&quot;\n    value {\n      i: 42\n    }\n  }\n  attr {\n    key: &quot;seed2&quot;\n    value {\n      i: 42\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform/mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;random_uniform/RandomUniform&quot;\n  input: &quot;random_uniform/sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;random_uniform&quot;\n  op: &quot;Add&quot;\n  input: &quot;random_uniform/mul&quot;\n  input: &quot;random_uniform/min&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;theta/Assign&quot;\n  op: &quot;Assign&quot;\n  input: &quot;theta&quot;\n  input: &quot;random_uniform&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@theta&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;validate_shape&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;init&quot;\n  op: &quot;NoOp&quot;\n  input: &quot;^theta/Assign&quot;\n}\nnode {\n  name: &quot;y&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 20640\n          }\n          dim {\n            size: 1\n          }\n        }\n        tensor_content: &quot;<stripped 82560 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;X&quot;\n  op: &quot;Const&quot;\n  attr {\n    key: &quot;dtype&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;value&quot;\n    value {\n      tensor {\n        dtype: DT_FLOAT\n        tensor_shape {\n          dim {\n            size: 20640\n          }\n          dim {\n            size: 9\n          }\n        }\n        tensor_content: &quot;<stripped 743040 bytes>&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;predictions&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;X&quot;\n  input: &quot;theta/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;sub&quot;\n  op: &quot;Sub&quot;\n  input: &quot;predictions&quot;\n  input: &quot;y&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;gradients/Square_grad/Mul&quot;\n  op: &quot;Mul&quot;\n  input: &quot;sub&quot;\n  input: &quot;gradients/Square_grad/Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;gradients/Square_grad/Mul_1&quot;\n  op: &quot;Mul&quot;\n  input: &quot;gradients/mse_grad/truediv&quot;\n  input: &quot;gradients/Square_grad/Mul&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;gradients/sub_grad/Neg&quot;\n  op: &quot;Neg&quot;\n  input: &quot;gradients/Square_grad/Mul_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;gradients/sub_grad/tuple/group_deps&quot;\n  op: &quot;NoOp&quot;\n  input: &quot;^gradients/Square_grad/Mul_1&quot;\n  input: &quot;^gradients/sub_grad/Neg&quot;\n}\nnode {\n  name: &quot;gradients/sub_grad/tuple/control_dependency_1&quot;\n  op: &quot;Identity&quot;\n  input: &quot;gradients/sub_grad/Neg&quot;\n  input: &quot;^gradients/sub_grad/tuple/group_deps&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@gradients/sub_grad/Neg&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/sub_grad/tuple/control_dependency&quot;\n  op: &quot;Identity&quot;\n  input: &quot;gradients/Square_grad/Mul_1&quot;\n  input: &quot;^gradients/sub_grad/tuple/group_deps&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@gradients/Square_grad/Mul_1&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;gradients/predictions_grad/MatMul_1&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;X&quot;\n  input: &quot;gradients/sub_grad/tuple/control_dependency&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: true\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;gradients/predictions_grad/MatMul&quot;\n  op: &quot;MatMul&quot;\n  input: &quot;gradients/sub_grad/tuple/control_dependency&quot;\n  input: &quot;theta/read&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;transpose_a&quot;\n    value {\n      b: false\n    }\n  }\n  attr {\n    key: &quot;transpose_b&quot;\n    value {\n      b: true\n    }\n  }\n}\nnode {\n  name: &quot;gradients/predictions_grad/tuple/group_deps&quot;\n  op: &quot;NoOp&quot;\n  input: &quot;^gradients/predictions_grad/MatMul&quot;\n  input: &quot;^gradients/predictions_grad/MatMul_1&quot;\n}\nnode {\n  name: &quot;gradients/predictions_grad/tuple/control_dependency_1&quot;\n  op: &quot;Identity&quot;\n  input: &quot;gradients/predictions_grad/MatMul_1&quot;\n  input: &quot;^gradients/predictions_grad/tuple/group_deps&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@gradients/predictions_grad/MatMul_1&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;GradientDescent/update_theta/ApplyGradientDescent&quot;\n  op: &quot;ApplyGradientDescent&quot;\n  input: &quot;theta&quot;\n  input: &quot;GradientDescent/learning_rate&quot;\n  input: &quot;gradients/predictions_grad/tuple/control_dependency_1&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@theta&quot;\n      }\n    }\n  }\n  attr {\n    key: &quot;use_locking&quot;\n    value {\n      b: false\n    }\n  }\n}\nnode {\n  name: &quot;GradientDescent&quot;\n  op: &quot;NoOp&quot;\n  input: &quot;^GradientDescent/update_theta/ApplyGradientDescent&quot;\n}\nnode {\n  name: &quot;gradients/predictions_grad/tuple/control_dependency&quot;\n  op: &quot;Identity&quot;\n  input: &quot;gradients/predictions_grad/MatMul&quot;\n  input: &quot;^gradients/predictions_grad/tuple/group_deps&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;_class&quot;\n    value {\n      list {\n        s: &quot;loc:@gradients/predictions_grad/MatMul&quot;\n      }\n    }\n  }\n}\nnode {\n  name: &quot;Square&quot;\n  op: &quot;Square&quot;\n  input: &quot;sub&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n}\nnode {\n  name: &quot;mse&quot;\n  op: &quot;Mean&quot;\n  input: &quot;Square&quot;\n  input: &quot;Const&quot;\n  attr {\n    key: &quot;T&quot;\n    value {\n      type: DT_FLOAT\n    }\n  }\n  attr {\n    key: &quot;Tidx&quot;\n    value {\n      type: DT_INT32\n    }\n  }\n  attr {\n    key: &quot;keep_dims&quot;\n    value {\n      b: false\n    }\n  }\n}\n';
          }
        </script>
        <link rel=&quot;import&quot; href=&quot;https://tensorboard.appspot.com/tf-graph-basic.build.html&quot; onload=load()>
        <div style=&quot;height:600px&quot;>
          <tf-graph-basic id=&quot;graph0.3745401188473625&quot;></tf-graph-basic>
        </div>
    "></iframe>
    


## Using TensorBoard
第一步是明确要可视化的图和你想可视化的参数

注意每次file_writer都要使用不同的日志，否则在读出来的时候会发生重叠乱成一团。最简单的方法如下，每次给文件名后面加上日期


```python
reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
```


```python
n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
```

第二步是创建节点

如下，第一行是创建节点计算MSE的值，并把它写成一个名为summary的二进制文件

第二行是创建一个FileWriter节点，可以向日志文件写summaries，它的两个参数分别是日志地址（创建events文件）和要可视化的图


```python
mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```


```python
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))
```

第三步是在execution phase里面，每隔一定的迭代次数求mse_summary节点的输出值，这个能输出一个summary我们要用file_writrer节点的功能把它写到日志events文件里。每隔多少step写一次


```python
with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()                                                     # not shown
```

最后一步，要记得关掉filer_writer


```python
file_writer.close()
```


```python
best_theta
```




    array([[ 2.07033372],
           [ 0.86371452],
           [ 0.12255151],
           [-0.31211874],
           [ 0.38510373],
           [ 0.00434168],
           [-0.01232954],
           [-0.83376896],
           [-0.80304712]], dtype=float32)



tensorboard打开方法见我的另一篇博文，在tensorflow分类下

在tensorboard中的Graphs选项下可看图的结构

当然还有别的类似tensorboard的图可视化包，比如
https://github.com/ericjang/tdb

这是个tensorflow debug包，没试过

# Name scopes

当图很复杂的时候，可以使用`with tf.name_scope("loss") as scope:`把节点nodes聚合起来。比如说把mse和error这两个ops聚合到loss namescope下面


```python
reset_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
```


```python
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
```


```python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```


```python
n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

file_writer.flush()
file_writer.close()
print("Best theta:")
print(best_theta)
```

    Best theta:
    [[ 2.07033372]
     [ 0.86371452]
     [ 0.12255151]
     [-0.31211874]
     [ 0.38510373]
     [ 0.00434168]
     [-0.01232954]
     [-0.83376896]
     [-0.80304712]]
    

可以发现这个ops（operation）现在在loss prefix下面


```python
print(error.op.name)
```

    loss/sub
    


```python
print(mse.op.name)
```

    loss/mse
    


```python
reset_graph()

a1 = tf.Variable(0, name="a")      # name == "a"
a2 = tf.Variable(0, name="a")      # name == "a_1"

with tf.name_scope("param"):       # name == "param"
    a3 = tf.Variable(0, name="a")  # name == "param/a"

with tf.name_scope("param"):       # name == "param_1"
    a4 = tf.Variable(0, name="a")  # name == "param_1/a"

for node in (a1, a2, a3, a4):
    print(node.op.name)
```

    a
    a_1
    param/a
    param_1/a
    

# Modularity 模块化
如果你想创建一个图计算两个ReLU的相加结果。如下的代码非常繁琐

An ugly flat code:


```python
reset_graph()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0., name="relu1")
relu2 = tf.maximum(z1, 0., name="relu2")  # Oops, cut&paste error! Did you spot it?

output = tf.add(relu1, relu2, name="output")
```

更好的方法是使用函数来模块化地计算relu

注意`tf.add_n()`是计算一个列表里tensor的和

Much better, using a function to build the ReLUs:

在下面这个cell里面，这些节点的名称会被命名为`weights`,`weights_1`,`weights_2`这样，如果发生重复，会自动给它的name后面加后缀显示在tensorboard的graph里面


```python
reset_graph()

def relu(X):
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X, w), b, name="z")
    return tf.maximum(z, 0., name="relu")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```


```python
file_writer = tf.summary.FileWriter("logs/relu1", tf.get_default_graph())
```

当然如果把它们都包在relu这个新定义的namescope就更好

Even better using name scopes:


```python
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
        b = tf.Variable(0.0, name="bias")                             # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
        return tf.maximum(z, 0., name="max")                          # not shown
```


```python
n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
file_writer.close()
```

# Sharing Variables 共用变量

## 像parameter一样传递进函数
Sharing a `threshold` variable the classic way, by defining it outside of the `relu()` function then passing it as a parameter:


```python
reset_graph()

def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")
```

## 把变量绑定为函数的一个属性


```python
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, relu.threshold, name="max")
```


```python
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")
```

## 使用tf.get_variable()方法创建和复用变量variable
创建和复用由with tf.variable_scope控制

如下代码，创建一个变量名为relu/threshold，但还没有指定复用。如果已经创建过了，则抛出一个错误


```python
reset_graph()

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
```

### 指定复用
以下两种方法都可以指定复用，如果还没有被创建则会抛出一个错误


```python
with tf.variable_scope("relu", reuse=True):
    threshold = tf.get_variable("threshold")
```


```python
with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")
```

完整的用法如下。定义函数并说明可复用，在函数外初始化指定的那个变量

使用get_variable()创建的variable都是以tf.variable_scope的名字作为前缀。如果这个前缀已经有重复了，那它会自动加上后缀。如relu_1/threshold和relu_2/threshold


```python
reset_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")
```


```python
file_writer = tf.summary.FileWriter("logs/relu6", tf.get_default_graph()) # relu6是文件名
file_writer.close()
```

有时需要把variable的定义放在relu函数外。（不是特别理解）

如下两个cell，都是第一个relu不指定复用，其他几个指定复用。这样shared variable是在第一个relu里面，其他几个relu复用了第一个relu分享的变量
（不同于上面，上面是relu每一个relu都复用了函数relu分享的变量）


```python
reset_graph()

def relu(X):
    with tf.variable_scope("relu"):
        threshold = tf.get_variable("threshold", shape=(), initializer=tf.constant_initializer(0.0))
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X, w), b, name="z")
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("", default_name="") as scope:
    first_relu = relu(X)     # create the shared variable
    scope.reuse_variables()  # then reuse it
    relus = [first_relu] + [relu(X) for i in range(4)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu8", tf.get_default_graph())
file_writer.close()
```


```python
reset_graph()

def relu(X):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
    b = tf.Variable(0.0, name="bias")                           # not shown
    z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")
```


```python
file_writer = tf.summary.FileWriter("logs/relu9", tf.get_default_graph())
file_writer.close()
```

# Extra material

## name_scope示例

The first `variable_scope()` block first creates the shared variable `x0`, named `my_scope/x`. For all operations other than shared variables (including non-shared variables), the variable scope acts like a regular name scope, which is why the two variables `x1` and `x2` have a name with a prefix `my_scope/`. Note however that TensorFlow makes their names unique by adding an index: `my_scope/x_1` and `my_scope/x_2`.

The second `variable_scope()` block reuses the shared variables in scope `my_scope`, which is why `x0 is x3`. Once again, for all operations other than shared variables it acts as a named scope, and since it's a separate block from the first one, the name of the scope is made unique by TensorFlow (`my_scope_1`) and thus the variable `x4` is named `my_scope_1/x`.

The third block shows another way to get a handle on the shared variable `my_scope/x` by creating a `variable_scope()` at the root scope (whose name is an empty string), then calling `get_variable()` with the full name of the shared variable (i.e. `"my_scope/x"`).这个variable_scope是一个空字符，说明它定义在root scope，需要call全名(name_scope + variable name)去get_variable


```python
reset_graph()

with tf.variable_scope("my_scope"):
    x0 = tf.get_variable("x", shape=(), initializer=tf.constant_initializer(0.))
    x1 = tf.Variable(0., name="x")
    x2 = tf.Variable(0., name="x")

with tf.variable_scope("my_scope", reuse=True):
    x3 = tf.get_variable("x")
    x4 = tf.Variable(0., name="x")

with tf.variable_scope("", default_name="", reuse=True):
    x5 = tf.get_variable("my_scope/x")

print("x0:", x0.op.name)
print("x1:", x1.op.name)
print("x2:", x2.op.name)
print("x3:", x3.op.name)
print("x4:", x4.op.name)
print("x5:", x5.op.name)
print(x0 is x3 and x3 is x5)
```

    x0: my_scope/x
    x1: my_scope/x_1
    x2: my_scope/x_2
    x3: my_scope/x
    x4: my_scope_1/x
    x5: my_scope/x
    True
    

## Strings 字符串
使用tf.constant定义


```python
reset_graph()

text = np.array("Do you want some café?".split())
text_tensor = tf.constant(text)

with tf.Session() as sess:
    print(text_tensor.eval())
```

    [b'Do' b'you' b'want' b'some' b'caf\xc3\xa9?']
    

## Autodiff

Note: the autodiff content was moved to the [extra_autodiff.ipynb](extra_autodiff.ipynb) notebook.

# Exercise solutions 一个完整的tensorflow逻辑回归流程

## 1. to 11.

See appendix A.

## 12. Logistic Regression with Mini-Batch Gradient Descent using TensorFlow

First, let's create the moons dataset using Scikit-Learn's `make_moons()` function:


```python
from sklearn.datasets import make_moons

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
```


```python
y_moons.shape, X_moons.shape
```




    ((1000,), (1000, 2))



Let's take a peek at the dataset:


```python
plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
plt.legend()
plt.show()
```


![png](output_158_0.png)


We must not forget to add an extra bias feature ($x_0 = 1$) to every instance. For this, we just need to add a column full of 1s on the left of the input matrix $\mathbf{X}$:


```python
X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
```

Let's check:


```python
X_moons_with_bias[:5]
```




    array([[ 1.        , -0.05146968,  0.44419863],
           [ 1.        ,  1.03201691, -0.41974116],
           [ 1.        ,  0.86789186, -0.25482711],
           [ 1.        ,  0.288851  , -0.44866862],
           [ 1.        , -0.83343911,  0.53505665]])



Looks good. Now let's reshape `y_train` to make it a column vector (i.e. a 2D array with a single column):


```python
y_moons_column_vector = y_moons.reshape(-1, 1)
```

Now let's split the data into a training set and a test set:


```python
test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:-test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]
```

随机创建batch，所以一般无法涵盖所有的example（instance），但这里简化了代码

Ok, now let's create a small function to generate training batches. In this implementation we will just pick random instances from the training set for each batch. This means that a single batch may contain the same instance multiple times, and also a single epoch may not cover all the training instances (in fact it will generally cover only about two thirds of the instances). However, in practice this is not an issue and it simplifies the code:


```python
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch
```

Let's look at a small batch:


```python
X_batch, y_batch = random_batch(X_train, y_train, 5)
X_batch
```




    array([[ 1.        ,  1.93189866,  0.13158788],
           [ 1.        ,  1.07172763,  0.13482039],
           [ 1.        , -1.01148674, -0.04686381],
           [ 1.        ,  0.02201868,  0.19079139],
           [ 1.        , -0.98941204,  0.02473116]])




```python
y_batch
```




    array([[1],
           [0],
           [0],
           [1],
           [0]])



Great! Now that the data is ready to be fed to the model, we need to build that model. Let's start with a simple implementation, then we will add all the bells and whistles.

First let's reset the default graph.


```python
reset_graph()
```

The _moons_ dataset has two input features, since each instance is a point on a plane (i.e., 2-Dimensional):


```python
n_inputs = 2
```

Now let's build the Logistic Regression model. As we saw in chapter 4, this model first computes a weighted sum of the inputs (just like the Linear Regression model), and then it applies the sigmoid function to the result, which gives us the estimated probability for the positive class:

$\hat{p} = h_\boldsymbol{\theta}(\mathbf{x}) = \sigma(\boldsymbol{\theta}^T \mathbf{x})$


Recall that $\boldsymbol{\theta}$ is the parameter vector, containing the bias term $\theta_0$ and the weights $\theta_1, \theta_2, \dots, \theta_n$. The input vector $\mathbf{x}$ contains a constant term $x_0 = 1$, as well as all the input features $x_1, x_2, \dots, x_n$.

Since we want to be able to make predictions for multiple instances at a time, we will use an input matrix $\mathbf{X}$ rather than a single input vector. The $i^{th}$ row will contain the transpose of the $i^{th}$ input vector $(\mathbf{x}^{(i)})^T$. It is then possible to estimate the probability that each instance belongs to the positive class using the following equation:

$ \hat{\mathbf{p}} = \sigma(\mathbf{X} \boldsymbol{\theta})$

That's all we need to build the model:


```python
X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
logits = tf.matmul(X, theta, name="logits")
y_proba = 1 / (1 + tf.exp(-logits))
```

In fact, TensorFlow has a nice function `tf.sigmoid()` that we can use to simplify the last line of the previous code:


```python
y_proba = tf.sigmoid(logits)
```

As we saw in chapter 4, the log loss is a good cost function to use for Logistic Regression:

$J(\boldsymbol{\theta}) = -\dfrac{1}{m} \sum\limits_{i=1}^{m}{\left[ y^{(i)} \log\left(\hat{p}^{(i)}\right) + (1 - y^{(i)}) \log\left(1 - \hat{p}^{(i)}\right)\right]}$

One option is to implement it ourselves:


```python
epsilon = 1e-7  # to avoid an overflow when computing the log
loss = -tf.reduce_mean(y * tf.log(y_proba + epsilon) + (1 - y) * tf.log(1 - y_proba + epsilon))
```

But we might as well use TensorFlow's `tf.losses.log_loss()` function:


```python
loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default
```

The rest is pretty standard: let's create the optimizer and tell it to minimize the cost function:


```python
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
```

All we need now (in this minimal version) is the variable initializer:


```python
init = tf.global_variables_initializer()
```

And we are ready to train the model and use it for predictions!

There's really nothing special about this code, it's virtually the same as the one we used earlier for Linear Regression:


```python
n_epochs = 1000
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        if epoch % 100 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)

    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
```

    Epoch: 0 	Loss: 0.792602
    Epoch: 100 	Loss: 0.343463
    Epoch: 200 	Loss: 0.30754
    Epoch: 300 	Loss: 0.292889
    Epoch: 400 	Loss: 0.285336
    Epoch: 500 	Loss: 0.280478
    Epoch: 600 	Loss: 0.278083
    Epoch: 700 	Loss: 0.276154
    Epoch: 800 	Loss: 0.27552
    Epoch: 900 	Loss: 0.274912
    

Note: we don't use the epoch number when generating batches, so we could just have a single `for` loop rather than 2 nested `for` loops, but it's convenient to think of training time in terms of number of epochs (i.e., roughly the number of times the algorithm went through the training set).

For each instance in the test set, `y_proba_val` contains the estimated probability that it belongs to the positive class, according to the model. For example, here are the first 5 estimated probabilities:


```python
y_proba_val[:5]
```




    array([[ 0.54895616],
           [ 0.70724374],
           [ 0.51900256],
           [ 0.9911136 ],
           [ 0.50859052]], dtype=float32)



To classify each instance, we can go for maximum likelihood: classify as positive any instance whose estimated probability is greater or equal to 0.5:

这个写得太巧妙了，好好学习一下！


```python
y_pred = (y_proba_val >= 0.5)
y_pred[:5]
```




    array([[ True],
           [ True],
           [ True],
           [ True],
           [ True]], dtype=bool)



Depending on the use case, you may want to choose a different threshold than 0.5: make it higher if you want high precision (but lower recall), and make it lower if you want high recall (but lower precision). See chapter 3 for more details.

Let's compute the model's precision and recall:


```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_test, y_pred)
```




    0.86274509803921573




```python
recall_score(y_test, y_pred)
```




    0.88888888888888884



Let's plot these predictions to see what they look like:


```python
y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()
```


![png](output_203_0.png)


上面那两行代码的意思，是根据y_pred，需要是True才能入选


```python
import numpy as np
y = np.array([[True, True]])
y_ = y.reshape(-1)
```

~是取反的意思，~x 类似于 -x-1 （涉及到负数的表示，补码，我还不太懂）


```python
~y
```




    array([[False, False]])




```python
Xtest = np.array([[ 1.        , -0.05146968,  0.44419863],
       [ 1.        ,  1.03201691, -0.41974116]])
Xtest[y_,1]
```




    array([-0.05146968])



Well, that looks pretty bad, doesn't it? But let's not forget that the Logistic Regression model has a linear decision boundary, so this is actually close to the best we can do with this model (unless we add more features, as we will show in a second).

下面要加入一些多项式的特征，并综合运用这一章学过的知识

Now let's start over, but this time we will add all the bells and whistles, as listed in the exercise:
* Define the graph within a `logistic_regression()` function that can be reused easily.
* Save checkpoints using a `Saver` at regular intervals during training, and save the final model at the end of training.
* Restore the last checkpoint upon startup if training was interrupted.
* Define the graph using nice scopes so the graph looks good in TensorBoard.
* Add summaries to visualize the learning curves in TensorBoard.
* Try tweaking some hyperparameters such as the learning rate or the mini-batch size and look at the shape of the learning curve.

Before we start, we will add 4 more features to the inputs: ${x_1}^2$, ${x_2}^2$, ${x_1}^3$ and ${x_2}^3$. This was not part of the exercise, but it will demonstrate how adding features can improve the model. We will do this manually, but you could also add them using `sklearn.preprocessing.PolynomialFeatures`.


```python
X_train_enhanced = np.c_[X_train,
                         np.square(X_train[:, 1]),
                         np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3,
                         X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test,
                        np.square(X_test[:, 1]),
                        np.square(X_test[:, 2]),
                        X_test[:, 1] ** 3,
                        X_test[:, 2] ** 3]
```

This is what the "enhanced" training set looks like:


```python
X_train_enhanced[:5]
```




    array([[  1.00000000e+00,  -5.14696757e-02,   4.44198631e-01,
              2.64912752e-03,   1.97312424e-01,  -1.36349734e-04,
              8.76459084e-02],
           [  1.00000000e+00,   1.03201691e+00,  -4.19741157e-01,
              1.06505890e+00,   1.76182639e-01,   1.09915879e+00,
             -7.39511049e-02],
           [  1.00000000e+00,   8.67891864e-01,  -2.54827114e-01,
              7.53236288e-01,   6.49368582e-02,   6.53727646e-01,
             -1.65476722e-02],
           [  1.00000000e+00,   2.88850997e-01,  -4.48668621e-01,
              8.34348982e-02,   2.01303531e-01,   2.41002535e-02,
             -9.03185778e-02],
           [  1.00000000e+00,  -8.33439108e-01,   5.35056649e-01,
              6.94620746e-01,   2.86285618e-01,  -5.78924095e-01,
              1.53179024e-01]])



Ok, next let's reset the default graph:


```python
reset_graph()
```

Now let's define the `logistic_regression()` function to create the graph. We will leave out the definition of the inputs `X` and the targets `y`. We could include them here, but leaving them out will make it easier to use this function in a wide range of use cases (e.g. perhaps we will want to add some preprocessing steps for the inputs before we feed them to the Logistic Regression model).


```python
def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_including_bias = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)
        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("save"):
            saver = tf.train.Saver()
    return y_proba, loss, training_op, loss_summary, init, saver
```

Let's create a little function to get the name of the log directory to save the summaries for Tensorboard:


```python
from datetime import datetime

def log_dir(prefix=""): # default 默认是空值
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)
```

Next, let's create the graph, using the `logistic_regression()` function. We will also create the `FileWriter` to save the summaries to the log directory for Tensorboard:


```python
n_inputs = 2 + 4
logdir = log_dir("logreg")

X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
```

可以检查上一个training session是不是中断了，如果是的话我们继续上一次的training session

At last we can train the model! We will start by checking whether a previous training session was interrupted, and if so we will load the checkpoint and continue training from the epoch number we saved. In this example we just save the epoch number to a separate file, but in chapter 11 we will see how to store the training step directly as part of the model, using a non-trainable variable called `global_step` that we pass to the optimizer's `minimize()` method.

You can try interrupting training to verify that it does indeed restore the last checkpoint when you start it again.


```python
n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_logreg_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_logreg_model"

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):# if the checkpoint file exists, restore the model and load the epoch number
        with open(checkpoint_epoch_path, "rb") as f: # "rb" means read + binary mode
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
        file_writer.add_summary(summary_str, epoch)
        if epoch % 500 == 0:
            print("Epoch:", epoch, "\tLoss:", loss_val)
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1)) # write in bianry

    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
    os.remove(checkpoint_epoch_path) # 当训练出最终模型之后，可以把epoch记录迭代到哪里的文件删掉
```

    Epoch: 0 	Loss: 0.629985
    Epoch: 500 	Loss: 0.161224
    Epoch: 1000 	Loss: 0.119032
    Epoch: 1500 	Loss: 0.0973292
    Epoch: 2000 	Loss: 0.0836979
    Epoch: 2500 	Loss: 0.0743758
    Epoch: 3000 	Loss: 0.0675021
    Epoch: 3500 	Loss: 0.0622069
    Epoch: 4000 	Loss: 0.0580268
    Epoch: 4500 	Loss: 0.054563
    Epoch: 5000 	Loss: 0.0517083
    Epoch: 5500 	Loss: 0.0492377
    Epoch: 6000 	Loss: 0.0471673
    Epoch: 6500 	Loss: 0.0453766
    Epoch: 7000 	Loss: 0.0438187
    Epoch: 7500 	Loss: 0.0423742
    Epoch: 8000 	Loss: 0.0410892
    Epoch: 8500 	Loss: 0.0399709
    Epoch: 9000 	Loss: 0.0389202
    Epoch: 9500 	Loss: 0.0380107
    Epoch: 10000 	Loss: 0.0371557
    

Once again, we can make predictions by just classifying as positive all the instances whose estimated probability is greater or equal to 0.5:


```python
y_pred = (y_proba_val >= 0.5)
```


```python
precision_score(y_test, y_pred)
```




    0.97979797979797978




```python
recall_score(y_test, y_pred)
```




    0.97979797979797978




```python
y_pred_idx = y_pred.reshape(-1) # a 1D array rather than a column vector
plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], 'go', label="Positive")
plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], 'r^', label="Negative")
plt.legend()
plt.show()
```


![png](output_229_0.png)


Now that's much, much better! Apparently the new features really helped a lot.

Try starting the tensorboard server, find the latest run and look at the learning curve (i.e., how the loss evaluated on the test set evolves as a function of the epoch number):

```
$ tensorboard --logdir=tf_logs
```

下面这个非常值得学习，自动控制随机调超参的方法（learning rate和batch size）

Now you can play around with the hyperparameters (e.g. the `batch_size` or the `learning_rate`) and run training again and again, comparing the learning curves. You can even automate this process by implementing grid search or randomized search. Below is a simple implementation of a randomized search on both the batch size and the learning rate. For the sake of simplicity, the checkpoint mechanism was removed.

关于reciprocal函数，偏左的数较多，右边较少

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.reciprocal.html#scipy.stats.reciprocal

The `reciprocal()` function from SciPy's `stats` module returns a random distribution that is commonly used when you have no idea of the optimal scale of a hyperparameter. See the exercise solutions for chapter 2 for more details. 


```python
from scipy.stats import reciprocal

n_search_iterations = 10

for search_iteration in range(n_search_iterations):
    batch_size = np.random.randint(1, 100)
    learning_rate = reciprocal(0.0001, 0.1).rvs(random_state=search_iteration)

    n_inputs = 2 + 4
    logdir = log_dir("logreg")
    
    print("Iteration", search_iteration)
    print("  logdir:", logdir)
    print("  batch size:", batch_size)
    print("  learning_rate:", learning_rate)
    print("  training: ", end="")

    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(
        X, y, learning_rate=learning_rate)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10001
    n_batches = int(np.ceil(m / batch_size))

    final_model_path = "./my_logreg_model_%d" % search_iteration

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
            file_writer.add_summary(summary_str, epoch)
            if epoch % 500 == 0:
                print(".", end="")

        saver.save(sess, final_model_path)

        print()
        y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
        y_pred = (y_proba_val >= 0.5)
        
        print("  precision:", precision_score(y_test, y_pred))
        print("  recall:", recall_score(y_test, y_pred))
```

    Iteration 0
      logdir: tf_logs/logreg-run-20170606195328/
      batch size: 19
      learning_rate: 0.00443037524522
      training: .....................
      precision: 0.979797979798
      recall: 0.979797979798
    Iteration 1
      logdir: tf_logs/logreg-run-20170606195605/
      batch size: 80
      learning_rate: 0.00178264971514
      training: .....................
      precision: 0.969696969697
      recall: 0.969696969697
    Iteration 2
      logdir: tf_logs/logreg-run-20170606195646/
      batch size: 73
      learning_rate: 0.00203228544324
      training: .....................
      precision: 0.969696969697
      recall: 0.969696969697
    Iteration 3
      logdir: tf_logs/logreg-run-20170606195730/
      batch size: 6
      learning_rate: 0.00449152382514
      training: .....................
      precision: 0.980198019802
      recall: 1.0
    Iteration 4
      logdir: tf_logs/logreg-run-20170606200523/
      batch size: 24
      learning_rate: 0.0796323472178
      training: .....................
      precision: 0.980198019802
      recall: 1.0
    Iteration 5
      logdir: tf_logs/logreg-run-20170606200726/
      batch size: 75
      learning_rate: 0.000463425058329
      training: .....................
      precision: 0.912621359223
      recall: 0.949494949495
    Iteration 6
      logdir: tf_logs/logreg-run-20170606200810/
      batch size: 86
      learning_rate: 0.0477068184194
      training: .....................
      precision: 0.98
      recall: 0.989898989899
    Iteration 7
      logdir: tf_logs/logreg-run-20170606200851/
      batch size: 87
      learning_rate: 0.000169404470952
      training: .....................
      precision: 0.888888888889
      recall: 0.808080808081
    Iteration 8
      logdir: tf_logs/logreg-run-20170606200932/
      batch size: 61
      learning_rate: 0.0417146119941
      training: .....................
      precision: 0.980198019802
      recall: 1.0
    Iteration 9
      logdir: tf_logs/logreg-run-20170606201026/
      batch size: 92
      learning_rate: 0.000107429229684
      training: .....................
      precision: 0.882352941176
      recall: 0.757575757576
    
