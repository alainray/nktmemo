import jax
import argparse
import neural_tangents as nt
import jax.numpy as jnp
from models import LeNet, MLP, CNN, model_dict, model_params
import tensorflow_datasets as tfds
import optax
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint
from utils import generate_binary_cross_entropy_loss_fn

import numpy as np

def binarize_labels(labels, threshold_class):
    return (labels <= threshold_class).astype(int)

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  loss_fn  = generate_binary_cross_entropy_loss_fn(model.apply, state, images, labels)

  
  '''def loss_fn(params):
    logits = model.apply({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits'''

  grad_fn = jax.value_and_grad(loss_fn)
  (loss, logits), grads = grad_fn(state.params)
  #print(len(logits), logits[0], logits[1])
  #print(logits[1][0].astype(float))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy

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

parser = argparse.ArgumentParser()

parser.add_argument("arch")                     # Architecture: fc, minialex
parser.add_argument("ds")                       # Dataset 
#parser.add_argument("seed")                     # Random seed (an int)
args = parser.parse_args()
#seed = int(args.seed)

dataset = args.ds    # 'mnist', 'cifar10', 'cifar100'

#ds = 2
model_key = jax.random.PRNGKey(10)
#data  = jax.random.normal(model_key, (ds, 28,28,3))
model = model_dict[args.arch](**model_params[args.arch])

train_ds, test_ds = get_datasets(dataset)

# Turn dataset into binary version
train_ds['label'] = binarize_labels(train_ds['label'],4)
test_ds['label'] = binarize_labels(test_ds['label'],4)

epochs = 5
rng, init_rng = jax.random.split(model_key)

dataset_dims = {'mnist': [1,28,28,1],'cifar10': [1,32,32,3], 'cifar100': [1,32,32,3]}

params = model.init(rng, jnp.ones(dataset_dims[dataset]))['params']
lr=0.001
tx = optax.sgd(lr, 0.9)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

print("Starting Training")
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(state, train_ds,
                                                    64,
                                                    input_rng)
    print(f"Train loss: {train_loss:.2f}, Train Acc: {100*train_accuracy:.2f}%")
    _, test_loss, test_accuracy = apply_model(state, test_ds['image'],
                                              test_ds['label'])
    print(f"Test loss: {test_loss:.2f}, Test Acc: {100*test_accuracy:.2f}%")
    #save_checkpoint("ckpts",state,epoch, prefix="ckpoint", keep_every_n_steps=1)
    # Checkpoint model


#print(model.apply(init_vars, data))