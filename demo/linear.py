from mxnet import autograd, nd
from IPython import display
from matplotlib import pyplot as plt
import random



# 打乱顺序，分批量返回
def data_iter(batch_size, feature, lables):
    num_example = len(feature)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_example)])
        yield feature.take(j), lables.take(j)


def linreg(X, w, b):
    return nd.dot(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2



# 梯度下降函数
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


# 求方程   Y=Xw+b+c 的值
if __name__ == '__main__':
    num_feature = 2
    num_sample = 10000
    true_feature = [3, -4.2]
    true_b = 2.1
    # 随机生成的X1和X2
    features = nd.random.normal(scale=1, shape=(num_sample, num_feature))
    # 相当于 nd.dot(features,nd.array([[3],[4.2]]))+true_b
    lables = true_feature[0] * features[:, 0] + true_feature[1] * features[:, 1] + true_b
    # lables 相当于计算后y的值
    lables += nd.random.normal(scale=0.01, shape=lables.shape)

    w = nd.random.normal(scale=0.01, shape=(num_feature, 1))
    b = nd.zeros(shape=(1,))
    # 创建梯度
    w.attach_grad()
    b.attach_grad()
    # 循环
    lr = 0.3
    num_epochs = 3
    net = linreg
    loss = squared_loss
    batch_size = 10
    print([w,b])
    for epoch in range(num_epochs):
        # 迭代周期
        for X, y in data_iter(batch_size,features,lables):
            with autograd.record():
                l=loss(net(X,w,b),y)#损失函数
            l.backward()#对小量数据进行求梯度
            sgd([w,b],lr,batch_size)#梯度下降，对损失函数下降方向迭代
        train_l=loss(net(features,w,b),lables)
        print("第 %d 次训练，误差为 %f" %(epoch,train_l.mean().asnumpy()))