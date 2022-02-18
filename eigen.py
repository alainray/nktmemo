import jax
import argparse
import jax.numpy as jnp
from models import model_dict, model_params
import tensorflow_datasets as tfds
import tensorflow as tf
import optax
from flax.training import train_state
from utils import handle_eigendata, calculate_ntk_matrix, timing, get_datasets, extract_experiment_data
import numpy as np
from os.path import isfile, join
from os import listdir
from flax.training.checkpoints import restore_checkpoint
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


# Handling arguments
parser = argparse.ArgumentParser()

parser.add_argument("checkpoint_folder")                     # Architecture: fc, minialex
parser.add_argument("seed")                     # Random seed (an int)
parser.add_argument("n_ntk_data")
parser.add_argument("n_eigen")
parser.add_argument("--bs", type=int, default=32)
args = parser.parse_args()
root = args.checkpoint_folder
seed = int(args.seed)
n_ntk_data = int(args.n_ntk_data)
n_eigen = int(args.n_eigen)
batch_size=args.bs


dataset_dims = {'mnist': [1,28,28,1],
                'fashion_mnist': [1,28,28,1],
                'cifar10': [1,32,32,3]}


# Setup model
lr = 0.001
tx = optax.sgd(lr, 0.9)

# Get all files 

files = [f for f in listdir(root) if isfile(join(root, f)) and "ckpoint" in f]
files.sort()

ntk_ds = None

for f in files:

    # Load checkpoint
    exp = extract_experiment_data(f)

    model = model_dict[exp['arch']](**model_params[exp['arch']])
    # Load checkpoint
    rng = jax.random.PRNGKey(10) # Doesn't matter
    params = model.init(rng, jnp.ones(dataset_dims[exp['dataset']]))['params']
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = restore_checkpoint(f, state, prefix="")
    # Load dataset

    if ntk_ds is None or exp['dataset'] != current_dataset:
        current_dataset = exp['dataset']
        print("Loading datasets...")
        train_ds, _ = get_datasets(exp['dataset'])
        ds, *_ = train_ds['image'].shape
        print("Creating NTK testing dataset...")
        key = jax.random.PRNGKey(128)
        index = jax.random.choice(key, ds, (n_ntk_data,), replace=False) 
        index = list(index)
        ntk_ds = train_ds['image'][index, ...]  
    # Calculate Empirical NTK
    print(f"Calculating NTK for {exp['arch']}-{exp['dataset']} using {n_ntk_data} data points for epoch {exp['epoch']}")
    ntk_matrix = calculate_ntk_matrix(model, ntk_ds, state)
    # Calculate Eigenvals and Vectors for NTK Matrix
    handle_eigendata(ntk_matrix, top_k_eigen=n_eigen, prefix=f"{exp['arch']}|{exp['dataset']}|{exp['epoch']}_{exp['seed']}_{n_ntk_data}")
  