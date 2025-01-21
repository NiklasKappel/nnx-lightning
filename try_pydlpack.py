import dlpack.tests
import jax
import torch
from dlpack import asdlpack

assert torch.cuda.is_available()

a1 = jax.numpy.array([[1, 2], [3, 4]])
t1 = torch.from_dlpack(asdlpack(a1))

t2 = torch.tensor([[5, 6], [7, 8]]).cuda()
a2 = jax.numpy.from_dlpack(asdlpack(t2))

dlpack.tests.run()
