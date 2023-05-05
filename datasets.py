# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  return lambda x: x


def get_dataset(config, additional_dim=None, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
      which equals the number of steps jitted together.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1
  # Create additional data dimension when jitting multiple steps together
  if additional_dim is None:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]



  def create_dataset(data_path):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)


    hf = h5py.File(data_path, 'r')
    data = hf['data']
    data = np.transpose(data, axes=(0,2,1))
    print(data.shape)
    
    ds = tf.data.Dataset.from_tensor_slices((data,))

    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(config.data.data_path)
  #test_ds = create_dataset('test')

  return train_ds
