import tensorflow as tf

_CSV_COLUMNS = ['buyhistory1', 'buyhistory2', 'buyhistory3']

_CSV_COLUMN_DEFAULTS = [[''], [''], ['']]


# 用户购买历史购买

# 返回数据
def train_input_fn():
    def parse_csv(line):
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个。record_defaults用于指明每一列的缺失值用什么填充
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        # 转为字典类型{'a':'b'}
        features = dict(zip(_CSV_COLUMNS, columns))
        lables = features.pop('buyhistory3')
        return features, lables

    dataset = tf.data.TextLineDataset('./data/wide_component_eval.csv').map(parse_csv).batch(10)

    iterator = dataset.make_one_shot_iterator()
    batch_features, labels = iterator.get_next()

    return batch_features, labels


def eval_input_fn():
    dataset = tf.data.TextLineDataset('./data/wide_component_test.csv')

    return dataset


buyhistory1 = tf.feature_column.categorical_column_with_hash_bucket(key='buyhistory1',
                                                                    hash_bucket_size=5)
buyhistory2 = tf.feature_column.categorical_column_with_hash_bucket(key='buyhistory2',
                                                                    hash_bucket_size=5)
# buyhistory3 = tf.feature_column.categorical_column_with_hash_bucket('buyhistory3',
#                                                                     hash_bucket_size=5)

base_column = [buyhistory1, buyhistory2]


# # or
# keywords_embedded = embedding_column(keywords, 16)
# columns = [keywords_embedded, ...]
# features = tf.parse_example(..., features=make_parse_example_spec(columns))
# dense_tensor = input_layer(features, columns)
# ```

#  定义特征类型，这里直接hash化

##测试wide组件，wide组件是一个连续特征的乘积，按照论文所说的就是：购买A则为1，否则0，然后计算乘积
def main():
    # 运行环境参数
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0},
                                      inter_op_parallelism_threads=5,
                                      intra_op_parallelism_threads=10))

    base_feature, lables = train_input_fn()

    # 模型
    model = tf.estimator.LinearClassifier(model_dir='./model/wide_compoent_test',
                                          feature_columns=base_column,
                                          config=run_config)

    model.train(input_fn=lambda: train_input_fn)
    results = model.evaluate(input_fn=train_input_fn)

    for key in sorted(results):
        print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    main()