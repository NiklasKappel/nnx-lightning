import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import jax

jax.config.update("jax_default_device", jax.devices()[-1])

import torch

torch.set_default_device(f"cuda:{torch.cuda.device_count() - 1}")

import dlpack.tests

dlpack.tests.run()
