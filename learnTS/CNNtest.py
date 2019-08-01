import tensorflow as tf
import pandas as pd

import argparse

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def load_data(y_name='Species'):
    # 获取数据
    train_paht = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_paht = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    # 训练用数组
    train = pd.read_csv(train_paht, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    # 测试用数组
    test = pd.read_csv(test_paht, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(feature,lables,batch_size):
    #把数据都放到dataset里面
    dataset=tf.data.Dataset.from_tensor_slices((dict(feature),lables))


def main(argv):
    #1：表示从第一个开始，后面缺省
    args = parser.parse_args(argv[1:])
    (train_x,train_y),(test_x,test_y)=load_data()
    #用于存放特征数组
    my_feature_columns=[]
    print("out",train_x.keys())
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    #建立训练模型 10*10
    classifier=tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # 两个隐藏单元，一层10个
        hidden_units=[10,10],
        # 模型必须选出3个分类
        n_classes=3)

    #训练模型
    classifier.train(
        # input_fn=lambda:
    )



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
