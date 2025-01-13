# nnx-lightning

An example of training a Flax NNX model with PyTorch data loaders and the Lightning trainer.

Todo:

- Is it bad that `LightningModule.log` requires a blocking call to wait for metric values because it can't collect JAX array futures?
- Should we convert data to numpy/JAX arrays directly, or to torch tensors first and to numpy/JAX arrays later?

Notes:

- Hyperparameters that are given to the LightningModule on initialization and used in the step functions must be passed explicitly through the JIT boundary (possibly in a `step_config` PyTree).
- Lightning likes to warn about data loaders that don't use multiple processors, but JAX likes to warn that using Python multiprocessing results in deadlocks.

See also:

- https://github.com/ludwigwinkler/JaxLightning
- https://github.com/Lightning-AI/pytorch-lightning/issues/20458
