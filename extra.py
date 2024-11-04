import jax
from functools import partial
import jax.numpy as jnp

@partial(jax.jit, static_argnames=['forward_fn', 'vocab_size', 'batch_size', 'block_size', 'max_new_tokens'])
def generate(variables, forward_fn, index_seq, rng_key, vocab_size, batch_size, block_size, max_new_tokens):
    batched_choice = jax.vmap(jax.random.choice)

    for i in range(max_new_tokens):
        if i % 20 == 0:
          print(i)
        index_cond = index_seq[:, -block_size:]
        logits = forward_fn(variables, index_cond)
        logits = logits[:, -1, :]
        probs = jax.nn.softmax(logits, axis=-1)
        rng_key, subkey = jax.random.split(rng_key)
        batched_key = subkey.reshape(1, -1)
        batched_key = jnp.repeat(batched_key, batch_size, axis=0)
        a = jnp.arange(vocab_size).reshape(1, -1)
        a = jnp.repeat(a, batch_size, axis=0)
        next_indexes = batched_choice(batched_key, a, p=probs)
        next_indexes = next_indexes.reshape(batch_size, -1)
        index_seq = jnp.concatenate([index_seq, next_indexes], axis=1)
    return index_seq

def get_stats(tokens, stats=None):
  if stats is None:
     stats={}
  for pair in zip(tokens, tokens[1:]):
    stats[pair] = stats.get(pair, 0)+1
  return stats

def merge(ids, pair, idx):
  new_ids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        new_ids.append(idx)
        i+=2
    else:
       new_ids.append(ids[i])
       i+=1
  return new_ids
     
