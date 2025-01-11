# nnx-lightning

An example of training a Flax NNX model with PyTorch data loaders and the Lightning trainer.

Todo:

- Is it bad that `LightningModule.log` requires a blocking call to wait for metric values because it can't collect JAX array futures?
- Should we convert data to numpy/JAX arrays directly, or to torch tensors first and to numpy/JAX arrays later?

See also:

- https://github.com/ludwigwinkler/JaxLightning
- https://github.com/Lightning-AI/pytorch-lightning/issues/20458
