import ml_collections


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 4
  training.n_iters = 1000001
  training.snapshot_freq = 2000
  training.log_freq = 100
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.n_jitted_steps = 1
  training.reduce_mean = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16
  sampling.samples_per_cosmo = 16
  sampling.batch_size = 4

  # evaluation
  #config.eval = evaluate = ml_collections.ConfigDict()
  #evaluate.begin_ckpt = 9
  #evaluate.end_ckpt = 26
  #evaluate.batch_size = 1024
  #evaluate.enable_sampling = False
  #evaluate.num_samples = 50000
  #evaluate.enable_loss = True
  #evaluate.enable_bpd = False
  #evaluate.bpd_dataset = 'test'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.data_path = 'trainH.h5'
  data.image_size = 8192
  data.num_cosmos = 100
  data.random_mask = False
  data.normalized = False
  data.mu_training = 0.#1.480879177506722e-08
  data.sigma_training =  1.#0.007186142245193263
  data.uniform_dequantization = False
  data.num_channels = 2



  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50.0
  model.num_scales = 2000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
