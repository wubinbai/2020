import keras
import numpy as np
import matplotlib.pyplot as plt
# Sequential按顺序构成的模型
from keras.models import Sequential 
# Dense全连接层
from keras.layers import Dense,Activation
from keras.optimizers import SGD

# 使用Numpy生成200个-0.5~0.5之间的值
x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)

# y_data= x_data**2 + noise 
y_data = np.square(x_data) + noise # 效果与上面一致

# 显示随机点
plt.scatter(x_data, y_data)
plt.show()

# 建立一个顺序模型
model = Sequential()
# 1-10-1: 加入一个隐藏层（10个神经元）：来拟合更加复杂的线性模型。添加激活函数，来计算函数的非线性

model.add(Dense(units=10, input_dim=1, activation='relu'))# 全连接层：输入一维数据，输出10个神经元
# model.add(Activation('tanh')) # 也可以直接在Dense里面加激活函数
model.add(Dense(units=1, activation='tanh')) # 全连接层：由于有上一层的添加，所以输入维度默认是10（可以不用写），输出1个值（要写）
# model.add(Activation('tanh'))


# 自定义优化器SDG , 学习率默认是0.01(太小，导致要迭代好多次才能较好的拟合数据)
sgd = SGD(lr=0.3)
model.compile(optimizer=sgd, loss='mse')

# 训练3000次数据
for step in range(3001):
    cost = model.train_on_batch(x_data, y_data)
    if step%500 == 0:
        print('cost: ',cost)
        
# x_data输入神经网络中，得到预测值y_pred
y_pred = model.predict(x_data)

# 显示随机点
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred,'r-', lw=3)
plt.show()
