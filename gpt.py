import jax
import jax.numpy as jnp
import optax
import numpy as np
import flax.linen as nn
from extra import generate

batch_size = 64
block_size =256
learning_rate = 3e-4
n_embd=384
n_head = 6
n_layer = 6
dropout = 0.2
rng_key = jax.random.PRNGKey(128)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab_size {vocab_size}")

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# print(encode("Hello"))
# print(decode(encode("Hello")))

data = np.array(encode(text))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

print(len(train_data), len(val_data))

def get_batch(key, split):
    key, subkey = jax.random.split(key)
    data = train_data if split=="train" else val_data
    ix = jax.random.randint(key=subkey, minval=0, maxval=len(data)-block_size, shape=(batch_size,))
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch(rng_key, "train")
print(xb.shape, yb.shape)

def masked_fill(mask, a, fill):
  return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))

class Head(nn.Module):
  head_size: int
    
  @nn.compact
  def __call__(self, x):
    B,T,C = x.shape
    key = nn.Dense(self.head_size, use_bias=False)
    k = key(x)
    query = nn.Dense(self.head_size, use_bias=False)
    q = query(x)

    wei = q @ k.transpose((0, -1, -2)) * C**-0.5
    tril = jnp.tril(jnp.ones((T, T),  dtype=bool))
    tril = jnp.repeat(tril[None, ...], repeats=B, axis=0)
    wei = masked_fill(tril, wei, -jnp.inf)
    wei = jax.nn.softmax(wei, axis=-1)

    value = nn.Dense(self.head_size, use_bias=False)
    v = value(x)
    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  num_heads: int
  head_size: int

  @nn.compact
  def __call__(self, x):
    multi_heads = [Head(self.head_size) for _ in range(self.num_heads)]
    proj = nn.Dense(n_embd)
    out = jnp.concatenate([h(x) for h in multi_heads],axis=-1)
    return proj(out)

class FeedForward(nn.Module):
  n_embd: int
  dropout: int

  @nn.compact
  def __call__(self, x):
    net = nn.Sequential([
        nn.Dense(4*self.n_embd),
        jax.nn.relu,
        nn.Dense(self.n_embd),
        nn.Dropout(self.dropout, deterministic=True)
    ])
    return net(x)


class Block(nn.Module):
  n_embd: int
  n_head: int
  dropout: int

  @nn.compact
  def __call__(self, x):
    head_size = self.n_embd // self.n_head
    sa = MultiHeadAttention(self.n_head, head_size)
    x = x + sa(nn.LayerNorm()(x))
    ffd = FeedForward(self.n_embd, self.dropout)
    x = x + ffd(nn.LayerNorm()(x))
    return x


class BigramLanguageModel(nn.Module):
    vocab_size: int
    n_embd: int
    block_size: int
    n_layer: int
    dropout: int
    
    @nn.compact
    def __call__(self, x):
        B,T = x.shape
        token_embedding_table = nn.Embed(num_embeddings=self.vocab_size, features=self.n_embd)
        positional_embedding_table = nn.Embed(num_embeddings=self.block_size, features=self.n_embd)
        tok_emb = token_embedding_table(x)
        pos_emb = positional_embedding_table(jnp.arange(T))
        x = tok_emb + pos_emb
        decoder_heads = [Block(self.n_embd, n_head=4, dropout=self.dropout) for _ in range(self.n_layer)]
        decoder_heads.append(nn.LayerNorm())
        blocks = nn.Sequential(decoder_heads)
        x = blocks(x)
        lm_head = nn.Dense(self.vocab_size)
        logits = lm_head(x)
        return logits
    

model = BigramLanguageModel(vocab_size, n_embd, block_size, n_layer, dropout)

variables = model.init(rng_key, xb)

optimizer = optax.adamw(learning_rate=learning_rate)
opt_state = optimizer.init(variables)

def loss_fn(variables, forward_fn, index_seq, labels):
    logits = forward_fn(variables, index_seq)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = loss.mean()
    return loss

#training loop
num_steps = 100

for i in range(num_steps):
  _, subkey = jax.random.split(rng_key)
  xb, yb = get_batch(subkey, "train")

  loss, grads = jax.value_and_grad(loss_fn)(variables, model.apply, xb, yb)
  updates, opt_state = optimizer.update(grads, opt_state, variables)
  variables = optax.apply_updates(variables, updates)

  print(f"Epoch: {i}, Loss: {loss :.4f}")


# generate
index_seq = jnp.zeros(shape=(1,1), dtype=jnp.uint16)
max_new_tokens = 100

generated_indices = generate(variables, model.apply, index_seq, rng_key, vocab_size, 1, block_size, max_new_tokens)
generated_indices = list(np.array(generated_indices[0]))
print("Generated text: ")
print(decode(generated_indices))



