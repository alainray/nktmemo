import jax
import jax.numpy as jnp
import optax
from flax.core import freeze
from scipy.sparse.linalg import svds
import neural_tangents as nt
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {te-ts:.2f} sec")
        return result
    return wrap


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
def calculate_ntk_matrix(model, data, state, ntk_bs=50):

    kernel_fn = nt.batch(
        nt.empirical_kernel_fn(model.apply, vmap_axes=0, implementation=1, trace_axes=()),
        batch_size=ntk_bs,
        device_count=-1,
        store_on_device=False,
    )
    # This will take a lot of time (data, model and number of samples dependent)...
    return kernel_fn(data, None, "ntk", freeze({'params': state.params}))

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