import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_input, num_output, num_hiddens = 784, 10, 56

W1 = nd.random.normal(scale=0.01, shape=(num_input, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_output))
b2 = nd.zeros(num_output)

params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


def relu(X):
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape(-1, num_input)
    H = relu(nd.dot(X, W1) + b1)
    return relu(nd.dot(H, W2) + b2)




if __name__ == '__main__':
    loss = gloss.SoftmaxCrossEntropyLoss()
    num_epochs, lr = 20, 0.6
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
                  params, lr)
