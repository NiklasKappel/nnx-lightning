import jax

jax.config.update("jax_default_device", jax.devices()[-1])

import torch

assert torch.cuda.is_available()
torch.set_default_device(f"cuda:{torch.cuda.device_count() - 1}")

from dlpack import asdlpack

initial_jax_array = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024))
initial_torch_tensor = torch.randn(1024, 1024)
print(initial_jax_array.device, initial_torch_tensor.device)

# a1 = jax.numpy.array([[1, 2], [3, 4]])
# t1 = torch.from_dlpack(asdlpack(a1))
# print(a1, a1.device)
# print(t1, t1.device)
#
# t2 = torch.tensor([[5, 6], [7, 8]]).cuda()
# a2 = jax.numpy.from_dlpack(asdlpack(t2))
# print(t2, t2.device)
# print(a2, a2.device)
