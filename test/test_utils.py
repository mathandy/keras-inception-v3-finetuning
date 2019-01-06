from __future__ import division, print_function, absolute_import
import numpy as np


def test_batches(data_shape=(5, 4, 3, 2)):
    # NOTE: this does not check the include_remainder=False case

    def assert_correct_number_of_examples(data):
        num_batched_examples = sum(len(batch) for batch in batches(data, bs))
        assert len(data) == num_batched_examples

    def assert_correct_total_sum(data):
        m, ex_shape = data.shape[0], data.shape[1:]
        assert np.sum(data) == sum(k*np.prod(ex_shape) for k in range(m))

    # generate some data of input `data_shape`
    # the 0th example will be all zeros, the 1th all 1's, etc
    num_examples = data_shape[0]
    example_shape = data_shape[1:]
    data = np.stack([k * np.ones(example_shape) for k in range(num_examples)])
    print(data.shape)
    batch_sizes = range(1, num_examples + 1)
    for bs in batch_sizes:
        try:
            assert_correct_total_sum(data)
            assert_correct_number_of_examples(data)
        except AssertionError:
            print('batch_size', bs)
            raise
    print("All good!")
test_batches()