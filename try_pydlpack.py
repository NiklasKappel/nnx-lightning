import jax

jax_device = jax.devices()[-1]
jax.config.update("jax_default_device", jax_device)

import jax.numpy as jnp
import torch

torch_device = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
torch.set_default_device(torch_device)

from timeit import timeit

import numpy as np
from dlpack import asdlpack

initial_jax_array = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
assert initial_jax_array.device == jax_device

initial_torch_tensor = torch.randn(1024, 1024, device=torch_device)
assert initial_torch_tensor.device == torch_device


def convert_to_torch_and_back(jax_array):
    torch_tensor = torch.from_numpy(np.asarray(jax_array)).cuda(torch_device)
    new_jax_array = jnp.asarray(torch_tensor, device=jax_device).block_until_ready()
    return new_jax_array


def convert_to_jax_and_back(torch_tensor):
    jax_array = jnp.asarray(torch_tensor, device=jax_device).block_until_ready()
    new_torch_tensor = torch.from_numpy(np.asarray(jax_array)).cuda(torch_device)
    return new_torch_tensor


def convert_to_torch_and_back_dlpack(jax_array):
    torch_tensor = torch.from_dlpack(asdlpack(jax_array))  # type: ignore
    new_jax_array = jnp.from_dlpack(asdlpack(torch_tensor)).block_until_ready()
    return new_jax_array


def convert_to_jax_and_back_dlpack(torch_tensor):
    jax_array = jnp.from_dlpack(asdlpack(torch_tensor)).block_until_ready()
    new_torch_tensor = torch.from_dlpack(asdlpack(jax_array))  # type: ignore
    return new_torch_tensor


a1 = convert_to_torch_and_back(initial_jax_array)
t1 = convert_to_jax_and_back(initial_torch_tensor)
a2 = convert_to_torch_and_back_dlpack(initial_jax_array)
t2 = convert_to_jax_and_back_dlpack(initial_torch_tensor)
assert a1.device == jax_device
assert t1.device == torch_device
assert a2.device == jax_device
assert t2.device == torch_device


def time_execution(func, initial_data, num_iterations, description):
    def wrapper():
        data = initial_data
        for _ in range(num_iterations):
            data = func(data)

    elapsed_time = timeit(wrapper, number=1)
    print(
        f"[{description}] Execution time for {num_iterations} iterations: {elapsed_time:.6f} seconds"
    )


num_iterations = 10000

time_execution(
    convert_to_torch_and_back,
    initial_jax_array,
    num_iterations,
    "convert_to_torch_and_back",
)
time_execution(
    convert_to_jax_and_back,
    initial_torch_tensor,
    num_iterations,
    "convert_to_jax_and_back",
)
time_execution(
    convert_to_torch_and_back_dlpack,
    initial_jax_array,
    num_iterations,
    "convert_to_torch_and_back_dlpack",
)
time_execution(
    convert_to_jax_and_back_dlpack,
    initial_torch_tensor,
    num_iterations,
    "convert_to_jax_and_back_dlpack",
)
