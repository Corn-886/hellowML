import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

import pandas as pd

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:-1]))

# 对数据进行标准化处理，即特征缩放，使数据落在某个区间
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 减去均值，再除以标准差
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后空值填0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# 特征转换，把字母的特征转为数字
all_features = pd.get_dummies(all_features, dummy_na=True)
# 转NDArray
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_lables = nd.array(train_data.SalePrice.values).reshape((-1, 1))

# 求方差
loss = gloss.L2Loss()


# 网络结构
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(64))
    net.add(nn.Dense(1))
    net.initialize()
    return net

#对数均方根误差
def log_rmse(net, features, labels):
    # 将小于1的值设置为1，取对数更加稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    #返回指定的批量数据
    train_iter = gdata.DataLoader(
        gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 这里使用Adam优化
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})

    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证,把样本分成若干份，返回第i份的训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    #第一维整除k
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        if i == 0:
            #画图的数据，
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
        print('fold %d ,train rmse %f ,valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size =  5, 60, 0.01, 0, 64
#num_epochs 迭代次数
train_l, valid_l = k_fold(k, train_features, train_lables, num_epochs, lr, weight_decay, batch_size)

print('%d -fold validation: avg train rmse %f ,avg valid rmse %f' % (k, train_l, valid_l))


def train_and_pred(train_features, test_features, train_lables, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_lables, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(train_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])


if __name__ == '__main__':
    train_and_pred(train_features, test_features, train_lables, test_data,
                   num_epochs, lr, weight_decay, batch_size)

#note:
# 关键参数：
# k, k折交叉验证分割成的份数
# num_epochs,每份数据训练次数
# lr,下降率，导数每次移动多少，最好注意不要收敛
# weight_decay,
# batch_size,批量大小，每次计算的数据样本大小
# net，神经网络隐藏层，通过调节目标函数使数据拟合
# 隐藏层及其输出个数
# 指标：
# 折线图，为什么会趋于平缓？如果波动太大可以调节lr下降率来减少步长，避免震荡
# rmse




