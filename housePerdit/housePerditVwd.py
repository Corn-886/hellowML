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
        model_dir=model_dir, feature_columns=get_base_column(), config=run_config,
        optimizer=lambda: tf.train.AdamOptimizer(learning_rate=0.01),
        activation_fn=tf.nn.relu,
        dropout=0.3,
        hidden_units=[64, 32],

        loss_reduction=tf.losses.Reduction.SUM
    )

    run_params = {
        'batch_size': 64,
        'train_epochs': 64,

    }
    for n in range(run_params.get('train_epochs')):
        model.train(input_fn=lambda: construct_input_fn(data_path=data_path, branch_size=run_params.get('batch_size'),
                                                        num_epochs=n, shuffle=True))
        results = model.evaluate(
            input_fn=lambda: construct_input_fn(data_path, 1, n, shuffle=False))
        print(results)

def apply_clean(model_dir):
    if tf.io.gfile.exists(model_dir):
        tf.compat.v1.logging.info("--clean flag set. Removing existing model dir:"
                                  " {}".format(model_dir))
        tf.io.gfile.rmtree(model_dir)
def main():
    model_dir = '/Users/suyuming/IdeaProjects/hellowML/housePerdit/model'
    data_paht = '/Users/suyuming/IdeaProjects/hellowML/housePerdit/data/train.csv'
    apply_clean(model_dir)
    run_loop(model_dir, data_paht)


if __name__ == '__main__':
    main()
