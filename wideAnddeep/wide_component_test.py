import tensorflow as tf

_CSV_COLUMNS = ['buyhistory1', 'buyhistory2', 'buyhistory3']
_CSV_COLUMN_DEFAULTS = [[''], [''], [0]]

_CSV_COLUMNS_TEST = ['buyhistory1', 'buyhistory2']
_CSV_COLUMN_DEFAULTS_TEST = [[''], ['']]


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

    dataset = tf.data.TextLineDataset('./data/wide_component_eval.csv').map(parse_csv).shuffle(10).batch(20)
    # dataset=dataset.shuffle(10)
    iterator = dataset.make_one_shot_iterator()
    batch_features, labels = iterator.get_next()

    return batch_features, labels


def eval_input_fn():
    # dataset = tf.data.TextLineDataset('./data/wide_component_test.csv')
    def parse_csv(line):
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS_TEST)
        features = dict(zip(_CSV_COLUMNS_TEST, columns))
        return features

    dataset = tf.data.TextLineDataset('./data/wide_component_test.csv').map(parse_csv).batch(3)

    iterator = dataset.make_one_shot_iterator()
    batch_features = iterator.get_next()
    return batch_features


buyhistory1 = tf.feature_column.categorical_column_with_hash_bucket(key='buyhistory1',
                                                                    hash_bucket_size=5)
buyhistory2 = tf.feature_column.categorical_column_with_hash_bucket(key='buyhistory2',
                                                                    hash_bucket_size=5)
categorical_feature_a_emb = tf.feature_column.embedding_column(
    categorical_column=buyhistory1, dimension=9)
categorical_feature_b_emb = tf.feature_column.embedding_column(
    categorical_column=buyhistory2, dimension=9)

base_DNN_column = [categorical_feature_a_emb, categorical_feature_b_emb]

base_column = [buyhistory1, buyhistory2]


#  定义特征类型，这里直接hash化

##测试wide组件，wide组件是一个连续特征的乘积，按照论文所说的就是：购买A则为1，否则0，然后计算乘积
def main():
    # 运行环境参数
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0},
                                      inter_op_parallelism_threads=5,
                                      intra_op_parallelism_threads=10))

    # 模型
    model = tf.estimator.LinearClassifier(model_dir='./model/wide_component_eval',
                                          feature_columns=base_column,
                                          n_classes=_CSV_COLUMNS.__len__(),
                                          config=run_config, optimizer=tf.train.FtrlOptimizer(
            learning_rate=0.02,
            l1_regularization_strength=1.0,
            l2_regularization_strength=1.0
        ))
    # model = tf.estimator.DNNClassifier(model_dir='./model/wide_component_eval_DNN',
    #                                    feature_columns=base_DNN_column,
    #                                    hidden_units=[ 10,10],
    #                                    activation_fn=tf.nn.sigmoid,

    #                                    config=run_config,
    #                                    n_classes=3)

    for n in range(50):
        model.train(input_fn=lambda: train_input_fn())
        results = model.evaluate(input_fn=train_input_fn)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))

        predictions = model.predict(
            input_fn=lambda: eval_input_fn())

        expected = ['buy_prd_A', 'buy_prd_b', 'buy_prd_c']
        for pred_dict, expec in zip(predictions, expected):
            class_id = pred_dict['class_ids'][0]
            print(class_id, pred_dict['probabilities'][class_id])
            # print(pred_dict)


if __name__ == '__main__':
    main()
