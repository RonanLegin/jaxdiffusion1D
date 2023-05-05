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
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
from models import ncsnpp1d
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import evaluation

import sde_lib
from absl import flags

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)

  optimizer = losses.get_optimizer(config)
  init_opt_state = optimizer.init(initial_params)

  state = mutils.State(step=0, opt_state=init_opt_state, lr=config.optim.lr,
                       model_state=init_model_state,
                       params=initial_params,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  tf.io.gfile.makedirs(checkpoint_dir)

  state = checkpoints.restore_checkpoint(checkpoint_dir, state)

  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build data iterators
  train_ds  = datasets.get_dataset(config,additional_dim=config.training.n_jitted_steps,uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimizer=optimizer,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)


  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays. Use ._numpy() to avoid copy.
    batch_xy = next(train_iter)  # pylint: disable=protected-access
    batch = {'data': jnp.array(scaler(batch_xy[0]))}
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    # Execute one training step
    (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss).mean()
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss))
      #writer.scalar("training_loss", loss, step)

    # Checkpoint
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf)
      # Generate and save samples
      if config.training.snapshot_sampling:
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        sample, n = sampling_fn(sample_rng, pstate)
        this_sample_dir = os.path.join(
          sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
        tf.io.gfile.makedirs(this_sample_dir)
        np.save(os.path.join(this_sample_dir, "sample.npy"), jnp.asarray(sample))

