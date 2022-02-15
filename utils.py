import jax
import jax.numpy as jnp
import optax
from flax.core import freeze
from scipy.sparse.linalg import svds
import neural_tangents as nt
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


def calculate_ntk_matrix(model, data, state):

    kernel_fn = nt.batch(
        nt.empirical_kernel_fn(model.apply, vmap_axes=0, implementation=1, trace_axes=()),
        batch_size=50,
        device_count=-1,
        store_on_device=False,
    )
    # This will take a lot of time (data, model and number of samples dependent)...
    return kernel_fn(data, None, "ntk", freeze({'params': state.params}))