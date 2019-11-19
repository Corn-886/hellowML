# 房屋预测 wide&deep 版本

from housePerdit.house_dataset import construct_input_fn
from housePerdit.house_dataset import get_base_column
import tensorflow as tf



def run_loop(model_dir, data_path):
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0},
                                      inter_op_parallelism_threads=5,
                                      intra_op_parallelism_threads=10
                                      ))

    model = tf.estimator.DNNRegressor(
        model_dir=model_dir, feature_columns=get_base_column(), config=run_config, optimizer=tf.train.AdamOptimizer(),
        activation_fn=tf.nn.sigmoid,
        dropout=0.3,
        hidden_units=[10,10],
        loss_reduction=tf.losses.Reduction.MEAN
    )
    run_params = {
        'batch_size': 10,
        'train_epochs': 5,
    }
    for n in range(5):
        model.train(input_fn=lambda: construct_input_fn(data_path, run_params.get('batch_size'), 100, True))
        results = model.evaluate(input_fn=lambda:construct_input_fn(data_path, run_params.get('batch_size'), 100, True))
        print(results)




def main():
    model_dir = '/Users/suyuming/IdeaProjects/hellowML/housePerdit/model'
    data_paht = '/Users/suyuming/IdeaProjects/hellowML/housePerdit/data/train.csv'
    run_loop(model_dir, data_paht)


if __name__ == '__main__':
    main()
