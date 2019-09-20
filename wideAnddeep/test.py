import pandas as pd
import json
import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]
_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def testJson():
    bufs = '{"holdstock":[{"name":"格力电器","code":"000651"},{"name":"贵州茅台","code":"600519"}],"manager":[],"managercom":[{"name":"易方达新经济","code":"易方达基金管理有限公司"},{"name":"易方达生物","code":"易方达基金管理有限公司"},{"name":"易方达新丝路","code":"易方达基金管理有限公司"},{"name":"易方达重组","code":"易方达基金管理有限公司"},{"name":"易方达消费行业","code":"易方达基金管理有限公司"},{"name":"创业ETF联接A","code":"易方达基金管理有限公司"}],"industri":"电子|家用电器|食品饮料","holdclass":"","holdcycle":""}'
    cls = json.loads(bufs)
    print(cls['holdstock'])
    for i in range(len(cls['holdstock'])):
        print(cls['holdstock'][i]['code'])


def input_fn(data_file, num_epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    assert tf.gfile.Exists(data_file), "{0} not found.".format(data_file)


    def parse_csv(line):
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个。record_defaults用于指明每一列的缺失值用什么填充
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')  # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise

    dataset = tf.data.TextLineDataset(data_file) \
        .map(parse_csv, num_parallel_calls=5)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'] + _NUM_EXAMPLES['validation'])

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def testColumnApi():
    # 定义特征为hash散列特征
    some_column_num = tf.feature_column.categorical_column_with_hash_bucket('product', hash_bucket_size=5)
    # 数据
    builder = _LazyBuilder({
        'product': [['a'], ['b']],
    })
    # 数据中特征信息的转化
    id_weight_pair = some_column_num._get_sparse_tensors(builder)

    with tf.Session() as sess:
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        print("稀疏矩阵：", id_tensor_eval)

        den_sedecord = tf.sparse_tensor_to_dense(id_tensor_eval, default_value=-1).eval(session=sess)
        print("稠密矩阵：", den_sedecord)


def main():
    # testColumnApi()
    print(_CSV_COLUMN_DEFAULTS.__len__())

# testJson()


if __name__ == '__main__':
    main()
