import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np


def load_data(path='./data/train.csv', test_split=0.2, seed=113):
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
    n_train = train_data.shape[0]
    train_features = np.array(all_features[:n_train].values)
    train_lables = np.array(train_data.SalePrice.values).reshape((-1, 1))




    np.random.seed(seed)
    indices = np.arange(len(train_features))
    np.random.shuffle(indices)

    x = train_features[indices]
    y = train_lables[indices]

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


(train_data, train_labels), (test_data, test_labels) = load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]


# def model_fn(features, labels, mode, params):
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation=tf.nn.relu,
#                            input_shape=(train_data.shape[1],)),
#         keras.layers.Dense(1)
#     ])


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.sigmoid),
        keras.layers.Dense(64, activation=tf.nn.sigmoid),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.AdamOptimizer()

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

def print_data():
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
    train_lables = np.array(train_data.SalePrice.values).reshape((-1, 1))
    eport_data=all_features+train_lables
    eport_data.to_csv('./a.csv')



def main():
    print_data()
    # model = build_model()
    #
    # history = model.fit(
    #     train_data, train_labels,
    #     epochs=1000, validation_split=0.2, verbose=0
    # )
    #
    # hist = pd.DataFrame(history.history)
    # hist['epoch'] = history.epoch
    # print(hist.tail(10))


if __name__ == '__main__':
    main()
