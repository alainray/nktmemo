import jax.numpy as jnp
import optax
from flax.core import freeze

def make_variables(params, model_state):
    return freeze({"params": params, **model_state})

def binary_cross_entropy_loss_with_logits(logits, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels[:, jnp.newaxis]))

def generate_binary_cross_entropy_loss_fn(apply_fn, state, images, labels):
    def loss_fn(params):
        #variables = make_variables(params, state)
        logits, new_model_state = apply_fn({"params": params}, images, mutable=["batch_stats"])
        loss = binary_cross_entropy_loss_with_logits(logits, labels)
        return loss, (new_model_state, logits)

    return loss_fn  