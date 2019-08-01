LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def main():
    print_tensors_in_checkpoint_file("/tmp/model/model.ckpt-56.index", tensor_name='b_point', all_tensors=False,
                                 all_tensor_names=False)

if __name__ == '__main__':
    main()
