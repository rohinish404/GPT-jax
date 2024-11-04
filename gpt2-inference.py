import jax
import jax.numpy as jnp
from functools import partial

def init_embedding(vocab_size, n_embd, key):
    return jax.random.normal(key, (vocab_size, n_embd), dtype=jnp.float16)

def init_layer_norm_params(n_embd):
    return {'gamma': jnp.ones((n_embd,)), 'beta': jnp.zeros((n_embd,))}

def init_mlp_params(n_embd, hidden_dim, key):
    k1, k2 = jax.random.split(key)
    return {
        'w1': jax.random.normal(k1, (n_embd, hidden_dim), dtype=jnp.float16),
        'b1': jnp.zeros((hidden_dim,), dtype=jnp.float16),
        'w2': jax.random.normal(k2, (hidden_dim, n_embd), dtype=jnp.float16),
        'b2': jnp.zeros((n_embd,), dtype=jnp.float16)
    }

def init_attention_params(n_embd, n_head, key):
    k_q, k_k, k_v, k_attn = jax.random.split(key, 4)
    head_dim = n_embd // n_head
    return {
        'w_q': jax.random.normal(k_q, (n_embd, head_dim * n_head), dtype=jnp.float16),
        'w_k': jax.random.normal(k_k, (n_embd, head_dim * n_head), dtype=jnp.float16),
        'w_v': jax.random.normal(k_v, (n_embd, head_dim * n_head), dtype=jnp.float16),
        'attn_linear': jax.random.normal(k_attn, (n_embd, n_embd), dtype=jnp.float16)
    }

def LayerNorm(params, x, eps=1e-5):
    xmean = jnp.mean(x, axis=-1, keepdims=True)
    xstd = jnp.std(x, axis=-1, keepdims=True)
    x_hat = (x - xmean) / (xstd + eps)
    return params['gamma'] * x_hat + params['beta']

def MLP(params, x):
    x = jnp.matmul(x, params['w1']) + params['b1']
    x = jax.nn.gelu(x, approximate='tanh')
    x = jnp.matmul(x, params['w2']) + params['b2']
    return x

def masked_fill(mask, a, fill_value):
    return jax.lax.select(mask, a, jax.lax.broadcast(fill_value, a.shape))

def CausalSelfAttention(params, x, n_head):
    B, T, C = x.shape
    head_dim = C // n_head

    q = jnp.matmul(x, params['w_q']).reshape(B, T, n_head, head_dim).transpose((0, 2, 1, 3))
    k = jnp.matmul(x, params['w_k']).reshape(B, T, n_head, head_dim).transpose((0, 2, 1, 3))
    v = jnp.matmul(x, params['w_v']).reshape(B, T, n_head, head_dim).transpose((0, 2, 1, 3))

    attn_weights = (q @ k.transpose((0, 1, 3, 2))) / jnp.sqrt(head_dim)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    mask = jnp.repeat(mask[None, ...], repeats=B, axis=0)
    mask = jnp.repeat(mask[None, ...], repeats=n_head, axis=1)
    attn_weights = masked_fill(mask, attn_weights, -jnp.inf)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)

    y = (attn_weights @ v).transpose((0, 2, 1, 3)).reshape(B, T, C)
    return jnp.matmul(y, params['attn_linear'])

def Block(params, x, n_head):
    # Attention block
    ln_1 = LayerNorm(params['ln1'], x)
    attn_out = CausalSelfAttention(params['attn'], ln_1, n_head)
    x = x + attn_out

    # MLP block
    ln_2 = LayerNorm(params['ln2'], x)
    mlp_out = MLP(params['mlp'], ln_2)
    x = x + mlp_out
    return x

def forward(params, input_ids, block_size, n_layer, n_head, n_embd):
    seq_len = input_ids.shape[1]
    wte, wpe = params['wte'], params['wpe'][:seq_len, :]

    # Embeddings
    x = wte[input_ids] + wpe

    # Transformer Blocks
    for i in range(n_layer):
        x = Block(params['blocks'][i], x, n_head)

    # Final LayerNorm and Output
    x = LayerNorm(params['ln_f'], x)
    logits = jnp.matmul(x, params['W_final'])
    return logits


def initialize_model_params(vocab_size, n_embd, n_head, n_layer, block_size, key):
    k_wte, k_wpe, k_final, *block_keys = jax.random.split(key, num=3 + n_layer)

    params = {
        'wte': init_embedding(vocab_size, n_embd, k_wte),
        'wpe': init_embedding(block_size, n_embd, k_wpe),
        'ln_f': init_layer_norm_params(n_embd),
        'W_final': jax.random.normal(k_final, (n_embd, vocab_size), dtype=jnp.float16),
        'blocks': []
    }

    for i, k in enumerate(block_keys):
        k_ln1, k_ln2, k_mlp, k_attn = jax.random.split(k, 4)
        block_params = {
            'ln1': init_layer_norm_params(n_embd),
            'attn': init_attention_params(n_embd, n_head, k_attn),
            'ln2': init_layer_norm_params(n_embd),
            'mlp': init_mlp_params(n_embd, 4 * n_embd, k_mlp)
        }
        params['blocks'].append(block_params)

    return params


def model_init(vocab_size, n_embd, n_head, n_layer, block_size, key):
    return initialize_model_params(vocab_size, n_embd, n_head, n_layer, block_size, key)

def model_forward(params, input_ids):
    block_size = params['wpe'].shape[0]
    n_layer = len(params['blocks'])
    n_embd = params['wte'].shape[1]
    n_head = params['blocks'][0]['attn']['w_q'].shape[1]
    return forward(params, input_ids, block_size, n_layer, n_head, n_embd)


