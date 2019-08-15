import d2lzh as d2l
from mxnet import autograd, nd

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input = 784
num_output = 10

W = nd.random.normal(scale=0.01, shape=(num_input, num_output))
b = nd.zeros(num_output)

W.attach_grad()
b.attach_grad()


# softmax函数，对矩阵数据进行e^x
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition


X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)


def net(X):
    return softmax(nd.dot(X.reshape((-1, num_input)), W) + b)


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


# 计算单个
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        y = y.astype('float32')
        # 取列最大值，看计算结果跟实际相差多远
        acc_sum += (net(X).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


num_epochs, lr = 15, 0.5


# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1)).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net)
        print("输出 %d,损失误差 %.4f 训练acc %.3f ,测试acc %.3f" % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        train_ls.append(train_l_sum / n)
        test_ls.append(test_acc)



if __name__ == '__main__':
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

    for X, y in test_iter:
        break

    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(X[0:9], titles[0:9])
