import jax
import argparse
import jax.numpy as jnp
from models import model_dict, model_params
import tensorflow_datasets as tfds
import tensorflow as tf
import optax
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint
from utils import handle_eigendata, calculate_ntk_matrix, timing
import numpy as np

def binarize_labels(labels, threshold_class):
    return (labels <= threshold_class).astype(int)

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
        preds = model.apply({'params': params}, images) 
        preds = preds.squeeze()
        loss = (- labels * jnp.log(preds+1e-6) - (1 - labels) * jnp.log(1 - preds + 1e-6)).mean()
        return loss, preds

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.greater(logits, 0.5*jnp.ones(logits.shape)) == labels)
  return grads, loss, accuracy

@timing
def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds['image'][perm, ...]
    batch_labels = train_ds['label'][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def get_datasets(ds_name, root_dir="."):
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder(ds_name)
  ds_builder.download_and_prepare(download_dir=root_dir)
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds

# Handling arguments
parser = argparse.ArgumentParser()

parser.add_argument("arch")                     # Architecture: fc, minialex
parser.add_argument("ds")                       # Dataset 
parser.add_argument("seed")                     # Random seed (an int)
parser.add_argument("epochs", type=int)
parser.add_argument("n_ntk_data")
parser.add_argument("n_eigen")
parser.add_argument("--bs", type=int, default=32)
args = parser.parse_args()
seed = int(args.seed)
n_ntk_data = int(args.n_ntk_data)
n_eigen = int(args.n_eigen)
dataset = args.ds    # 'mnist', 'cifar10', "fashion_mnist"
batch_size=args.bs

# Setting up model and data

model_key = jax.random.PRNGKey(seed)
model = model_dict[args.arch](**model_params[args.arch])
print("Loading datasets...")
train_ds, test_ds = get_datasets(dataset)
ds, *_ = train_ds['image'].shape
print("Creating NTK testing dataset...")
key = jax.random.PRNGKey(128)
index = jax.random.choice(key, ds, (n_ntk_data,), replace=False) 
index = list(index)
ntk_ds = train_ds['image'][index, ...]  
# Turn dataset into binary version
train_ds['label'] = binarize_labels(train_ds['label'],4)
test_ds['label'] = binarize_labels(test_ds['label'],4)

epochs = int(args.epochs)
rng, init_rng = jax.random.split(model_key)

dataset_dims = {'mnist': [1,28,28,1],
                'fashion_mnist': [1,28,28,1],
                'cifar10': [1,32,32,3]}

# Setup model
params = model.init(rng, jnp.ones(dataset_dims[dataset]))['params']
lr=0.001
tx = optax.sgd(lr, 0.9)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
epoch = 0
# Calculate Empirical NTK
ntk_matrix = calculate_ntk_matrix(model, ntk_ds, state)
# Calculate Eigenvals and Vectors for NTK Matrix

handle_eigendata(ntk_matrix, top_k_eigen=n_eigen, prefix=f"{args.arch}|{dataset}|{epoch}_{seed}")
# Checkpoint parameters
save_checkpoint("ckpts",state,"",
  prefix=f"ckpoint_{args.arch}|{dataset}|{epoch}_{seed}",
  keep_every_n_steps=1,
  overwrite=True)

print("Starting Training")
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    batch_size,
                                                    input_rng)
    print(f"Train loss: {train_loss:.2f}, Train Acc: {100*train_accuracy:.2f}%")
    _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                              test_ds['label'])
    print(f"Test loss: {test_loss:.2f}, Test Acc: {100*test_accuracy:.2f}%")

    stats = {"train": {"loss": train_loss, "acc": train_accuracy},
    "test": {"loss": test_loss, "acc": test_accuracy}}

    jnp.save(f"stats/{args.arch}|{dataset}|{epoch}_{seed}", stats)
    # Calculate Empirical NTK
    ntk_matrix = calculate_ntk_matrix(model, ntk_ds, state)
    # Calculate Eigenvals and Vectors for NTK Matrix
    handle_eigendata(ntk_matrix, top_k_eigen=n_eigen, prefix=f"{args.arch}|{dataset}|{epoch}_{seed}")
    # Checkpoint parameters
    save_checkpoint("ckpts",state,"",
      prefix=f"ckpoint_{args.arch}|{dataset}|{epoch}_{seed}",
      keep_every_n_steps=1,
      overwrite=True)
  