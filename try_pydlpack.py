import jax

jax_device = jax.devices()[-1]
jax.config.update("jax_default_device", jax_device)

import jax.numpy as jnp

import torch

assert torch.cuda.is_available()
torch_device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
torch.set_default_device(torch_device)

import numpy as np
import timeit
from dlpack import asdlpack

initial_jax_array = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
assert initial_jax_array.device == jax_device

initial_torch_tensor = torch.randn(1024, 1024, device=torch_device)
assert initial_torch_tensor.device == torch_device


def convert_to_torch_and_back(jax_array):
    torch_tensor = torch.from_numpy(np.asarray(jax_array)).cuda(torch_device)
    jax_array = jnp.asarray(torch_tensor, device=jax_device).block_until_ready()


def convert_to_jax_and_back(torch_tensor):
    jax_array = jnp.asarray(torch_tensor, device=jax_device).block_until_ready()
    torch_tensor = torch.from_numpy(np.asarray(jax_array)).cuda(torch_device)


def convert_to_torch_and_back_dlpack(jax_array):
    torch_tensor = torch.from_dlpack(asdlpack(jax_array))  # type: ignore
    jax_array = jnp.from_dlpack(asdlpack(torch_tensor)).block_until_ready()


def convert_to_jax_and_back_dlpack(torch_tensor):
    jax_array = jnp.from_dlpack(asdlpack(torch_tensor)).block_until_ready()
    torch_tensor = torch.from_dlpack(asdlpack(jax_array))  # type: ignore


convert_to_torch_and_back(initial_jax_array)
convert_to_jax_and_back(initial_torch_tensor)
convert_to_torch_and_back_dlpack(initial_jax_array)
convert_to_jax_and_back_dlpack(initial_torch_tensor)
