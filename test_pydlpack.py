import jax

jax.config.update("jax_default_device", jax.devices()[-1])

import torch

assert torch.cuda.is_available()
torch.set_default_device(f"cuda:{torch.cuda.device_count() - 1}")

import dlpack.tests

dlpack.tests.run()
