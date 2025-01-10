import warnings

warnings.filterwarnings(
    "ignore",
    "You are using `torch.load` with `weights_only=False`",
)
warnings.filterwarnings(
    "ignore",
    "TypedStorage is deprecated",
)
warnings.filterwarnings(
    "ignore",
    "`LightningModule.configure_optimizers` returned `None`",
)
warnings.filterwarnings(
    "ignore",
    ".*Consider increasing the value of the `num_workers` argument",
)

import jax
import lightning as L
import numpy as np
import optax
from flax import nnx
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils import data
from torchvision import datasets
from torchvision.transforms.functional import to_tensor


class CNN(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.avg_pool = nnx.avg_pool
        self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x):
        x = self.avg_pool(nnx.relu(self.conv1(x)), window_shape=(2, 2), strides=(2, 2))
        x = self.avg_pool(nnx.relu(self.conv2(x)), window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class LitCNN(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.005,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.model = CNN(rngs=nnx.Rngs(0))

    def training_step(self, batch):
        # Increment the global step counter.
        # See: https://github.com/ludwigwinkler/JaxLightning/issues/1
        self.optimizers().step()  # type: ignore

        loss = self.do_training_step(self.model, self.optimizer, batch)
        self.log("train_loss", loss.item())

    def validation_step(self, batch):
        loss = self.do_validation_step(self.model, batch)
        batch_size = batch[0].shape[0]
        self.log("val_loss", loss.item(), batch_size=batch_size)

    def configure_optimizers(self):
        # We create the optimizer in the `configure_optimizers` method so that
        # lightning can decide when to create or reset the optimizer.
        # We set the optimizer as an attribute so that we can access it manually.
        self.optimizer = nnx.Optimizer(
            self.model, optax.adamw(self.learning_rate, self.momentum)
        )
        # We (implicitly) return `None` because we don't have a PyTorch optimizer.

    @staticmethod
    def loss_fn(model: CNN, batch):
        logits = model(batch[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch[1]
        ).mean()
        return loss, logits

    @staticmethod
    @nnx.jit
    def do_training_step(model: CNN, optimizer: nnx.Optimizer, batch):
        grad_fn = nnx.value_and_grad(LitCNN.loss_fn, has_aux=True)
        (loss, _), grads = grad_fn(model, batch)
        optimizer.update(grads)
        return loss

    @staticmethod
    @nnx.jit
    def do_validation_step(model: CNN, batch):
        print("Compiling eval_step")
        loss, _ = LitCNN.loss_fn(model, batch)
        return loss


def get_loaders():
    # Here, the dataset returns inputs as PIL images and labels as torch
    # tensors. Using `transform`, the PIL images are converted to torch tensors
    # every time a sample is accessed. Using `numpy_collate`, the torch tensors
    # for inputs and labels are collated and then converted to numpy arrays.
    # The collated batch is then converted to JAX arrays in the
    # `do_training_step` and `do_validation_step` methods.

    # Ideally, the dataset would return numpy or JAX arrays directly so that
    # all conversions could be avoided.

    def transform(x):
        # Convert to tensor and move the feature dimension to the end.
        return to_tensor(x).movedim(0, -1)

    train_set = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=transform
    )
    valid_set = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=transform
    )

    def numpy_collate(batch):
        return jax.tree.map(np.asarray, data.default_collate(batch))

    batch_size = 32
    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=numpy_collate,
    )
    valid_loader = data.DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=numpy_collate,
    )

    return train_loader, valid_loader


def main():
    model = LitCNN()

    train_loader, valid_loader = get_loaders()

    logger = TensorBoardLogger("tb_logs")
    trainer = L.Trainer(
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=10,
        limit_val_batches=10,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=10,
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
