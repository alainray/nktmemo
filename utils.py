import jax
import jax.numpy as jnp
from flax.core import freeze
from scipy.sparse.linalg import svds
import neural_tangents as nt
from functools import wraps
from time import time
import tensorflow_datasets as tfds


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te-ts:.2f} sec")
        return result
    return wrap


def get_datasets(ds_name, root_dir="."):
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder(ds_name)
  ds_builder.download_and_prepare(download_dir=root_dir)
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = jnp.float32(train_ds['image']) / 255.
  test_ds['image'] = jnp.float32(test_ds['image']) / 255.
  return train_ds, test_ds


def make_variables(params, model_state):
    return freeze({"params": params, **model_state})



def ntk_eigenstuff(ntk_mat, top_k_eigen=100):
    ds,_,_, n_classes = ntk_mat.shape
    # Reshape to create appropriate matrix for SVD
    ntk_mat = jnp.reshape(ntk_mat,(ds,ds*n_classes*n_classes))
    # Get eigenvalues and eigenvectors
    _, eigvals, eigvecs = svds(jax.device_get(ntk_mat), 
                            k=top_k_eigen, 
                            return_singular_vectors=True)
    return eigvals, eigvecs

@timing
def calculate_ntk_matrix(model, data, params, ntk_bs=50):

    kernel_fn = nt.batch(
        nt.empirical_kernel_fn(model.apply, vmap_axes=0, implementation=1, trace_axes=()),
        batch_size=ntk_bs,
        device_count=-1,
        store_on_device=False,
    )
    # This will take a lot of time (data, model and number of samples dependent)...
    return kernel_fn(data, None, "ntk", freeze({'params': params}))

@timing
def handle_eigendata(ntk_mat, top_k_eigen=100, save_path="eigen", prefix=""):
    e_vals, e_vecs = ntk_eigenstuff(ntk_mat, top_k_eigen=top_k_eigen)
    total_sum = ntk_mat.trace()
    val_sum = e_vals.sum()
    ratio = float(val_sum/total_sum)
    jnp.save(f"{save_path}/eigvecs_{prefix}.npy", e_vecs)
    jnp.save(f"{save_path}/eigvals_{prefix}.npy", e_vals)
    jnp.save(f"{save_path}/trace_{prefix}.npy", total_sum)

    print(f"Top {top_k_eigen} eigenvalues represent: {100*ratio:.2f}%")

def extract_experiment_data(filename):
    filename = filename.replace("ckpoint_","")
    filename = filename.replace(".npy","")
    filename = filename.split("|")
    dataset = filename[1]
    arch = filename[0]
    epoch = int(filename[2].split("_")[0])
    seed = int(filename[2].split("_")[1])

    return {'dataset': dataset, 'arch': arch, 'epoch': epoch, "seed": seed}