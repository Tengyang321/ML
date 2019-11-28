#线性回归最小二乘法代数形式实现房价预测

#导入模块
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#x代表面积，y代表真实的房价
x = np.array([56, 72, 69, 88, 102, 86, 76, 79, 94, 74])
y = np.array([92, 102, 86, 110, 130, 99, 96, 102, 105, 92])

#定义拟合曲线
def f(x,b,w1):
    y = b + w1* x
    return  y

#定义损失函数
def square_loss(x, y, b, w1):
    loss = sum(np.square(y - (b + w1*x)))
    return loss

#平方损失函数最小时对应的w参数值 ，b
def get_b_w1(x, y):
    n = len(x)
    w1 = (n*sum(y*(x-sum(x)/n)))/(n*sum(x*x)-sum(x)*sum(x))
    b =sum(y-w1*x)/ n
    return w1, b

#测试
w1,b = get_b_w1(x, y)
#损失值
cost = square_loss(x, y, b, w1)
print("total cost is :",cost)
print(square_loss)
#带入线段的点
x_temp = np.linspace(50, 120, 100)
plt.scatter(x, y)
plt.plot(x_temp, x_temp * w1 + b, 'r')
plt.show()
#数值预测
f(120, b, w1)

