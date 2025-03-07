import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional
import flax.linen as nn
import optax
import math

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def polar(rho, theta):
  return rho * (jnp.cos(theta) + 1j * jnp.sin(theta))

def view_as_complex(x):
    return x[..., 0] + 1j * x[..., 1]

def view_as_real(x):
    return jnp.stack((jnp.real(x), jnp.imag(x)), axis=-1)

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must me divisible by 2"
    
    theta_numerator = jnp.arange(0, head_dim, 2).astype(jnp.float32)
    print(f"theta num {theta_numerator.shape}") #(Head dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))
    print(f"theta {theta.shape}") #(Head dim/2)

    m = jnp.arange(seq_len)
    print(f"m {m.shape}") #(SeqLen)
    freqs = jnp.outer(m, theta).astype(jnp.float32)
    print(f"freqs {freqs.shape}") #(SeqLen, HeadDim/2)

    ones = jnp.ones_like(freqs)
    freqs_complex = polar(ones, freqs)
    print(f"freqs_complex {freqs_complex.shape}") #(SeqLen, HeadDim/2)
    return freqs_complex

def apply_rotary_embeddings(x, freqs_complex):
    print(f"rotary embed x {x.shape}") # (B, Seqlen, H, Headdim)
    x_complex = view_as_complex(jnp.reshape(x.astype(jnp.float32),(*x.shape[:-1], x.shape[-1]//2, 2)))
    print(f"x_complex {x_complex.shape}") # (B, Seqlen, H, Headdim/2)


    freqs_complex = jnp.expand_dims(jnp.expand_dims(freqs_complex, 0), 2)
    print(f"freqs complex {freqs_complex.shape}") # (1, Seqlen, 1, Headdim/2)

    x_rotated = x_complex * freqs_complex
    print(f"x rotated {x_rotated.shape}") # (B, Seqlen, H, Headdim/2)

    x_out = view_as_real(x_rotated)
    print(f"x_out {x_out.shape}") # (B, Seqlen, H, Headdim/2, 2)

    x_out = jnp.reshape(x_out, (x.shape))
    print(f"x_out {x_out.shape}") # (B, Seqlen, H, Headdim)

    return x_out.astype(x.dtype)


class RMSNorm(nn.Module):
    dim: int
    eps: int

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.mean(jnp.power(x, 2), axis=-1, keepdims=True) + self.eps)

    @nn.compact
    def __call__(self, x):
        weight = jnp.ones(self.dim)
        return weight * self._norm(x)
    
def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return jnp.reshape(jnp.broadcast_to(x[:,:,:,None,:], (batch_size, seq_len, n_kv_heads, n_rep, head_dim)), (batch_size, seq_len, n_kv_heads* n_rep, head_dim))
    

class SelfAttention(nn.Module):
    config: ModelArgs

    @nn.compact
    def __call__(self, x, start_pos, freqs_complex):
        n_kv_heads = self.config.n_heads if self.config.n_kv_heads is None else self.config.n_kv_heads
        n_heads_q  = self.config.n_heads
        n_rep = n_heads_q // n_kv_heads
        n_kv_heads = self.config.n_heads if self.config.n_kv_heads is None else self.config.n_kv_heads
        head_dim = self.config.dim // self.config.n_heads
        wq = nn.Dense(self.config.n_heads * head_dim, use_bias=False)
        wk = nn.Dense(n_kv_heads * head_dim, use_bias=False)  # Use the local n_kv_heads variable
        wv = nn.Dense(n_kv_heads * head_dim, use_bias=False)
        wo = nn.Dense(self.config.dim, use_bias=False)

        cache_k = jnp.zeros((self.config.max_batch_size, self.config.max_seq_len, n_kv_heads, head_dim))
        cache_v = jnp.zeros((self.config.max_batch_size, self.config.max_seq_len, n_kv_heads, head_dim))

        batch_size, seq_len, _ = x.shape
        print(f"x shape attn {x.shape}") # (B, 1, Dim)
        
        xq = wq(x)
        print(f"xq {xq.shape}")# (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xk = wk(x)
        print(f"xk {xk.shape}")# (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = wv(x)
        print(f"xv {xv.shape}")# (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)

        xq = jnp.reshape(xq, (batch_size, seq_len, n_heads_q, head_dim))
        print(f"xq {xq.shape}")# (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xk = jnp.reshape(xk, (batch_size, seq_len, n_kv_heads, head_dim))
        print(f"xk {xk.shape}")# (B, 1, H_Q * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = jnp.reshape(xv, (batch_size, seq_len, n_kv_heads, head_dim))
        print(f"xv {xv.shape}")# (B, 1, H_Q * Head_Dim) -> (B, 1, H_KV, Head_Dim)

        xq = apply_rotary_embeddings(xq, freqs_complex)
        xk = apply_rotary_embeddings(xk, freqs_complex)

        cache_k = cache_k.at[:batch_size, start_pos:start_pos+seq_len].set(xk)
        cache_v = cache_v.at[:batch_size, start_pos:start_pos+seq_len].set(xv)

        keys = cache_k[:batch_size, 0:start_pos+seq_len]
        values = cache_v[:batch_size, 0:start_pos+seq_len]

        keys = repeat_kv(keys, n_rep)
        values =  repeat_kv(values, n_rep)

        xq = jnp.transpose(xq, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))

        scores = jnp.matmul(xq, jnp.transpose(keys, (0, 1, 3, 2))) / math.sqrt(head_dim)
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(xq.dtype)

        output = jnp.matmul(scores, values)
        output = jnp.reshape(jnp.transpose(output, (0,2,1,3)), (batch_size, seq_len, -1))

        return wo(output)


class FeedForward(nn.Module):
    config: ModelArgs

    @nn.compact
    def __call__(self, x):
        hidden_dim = 4 * self.config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if self.config.ffn_dim_multiplier is not None:
            hidden_dim = int(self.config.ffn_dim_multiplier * hidden_dim)
        
        hidden_dim = self.config.multiple_of * ((hidden_dim+ self.config.multiple_of - 1) // self.config.multiple_of)

        w1 = nn.Dense(hidden_dim, use_bias=False)
        w2 = nn.Dense(self.config.dim, use_bias=False)
        w3 = nn.Dense(hidden_dim, use_bias=False)

        swish = jax.nn.silu(w1(x))
        x_V = w3(x)
        x = swish * x_V
        x = w2(x)
        return x

class EncoderBlock(nn.Module):
    config: ModelArgs

    @nn.compact
    def __call__(self, x, start_pos, freqs_complex):
        attention = SelfAttention(self.config)
        feed_forward = FeedForward(self.config)

        attention_norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        ffn_norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)

        h = x + attention(attention_norm(x), start_pos, freqs_complex)
        out = h + feed_forward(ffn_norm(x))
        return out


class Transformer(nn.Module):
    config: ModelArgs

    @nn.compact
    def __call__(self, x, start_pos):
        assert self.config.vocab_size != -1, "Vocab size must be set"

        tok_embeddings = nn.Embed(num_embeddings=self.config.vocab_size, features=self.config.dim)
        layers = [EncoderBlock(self.config) for _ in range(self.config.n_layers)]
        norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        output = nn.Dense(self.config.vocab_size, use_bias=False)

        freqs_complex = precompute_theta_pos_frequencies(self.config.dim // self.config.n_heads, self.config.max_seq_len * 2)

        # (B, seq_len)
        print(f"x shape {x.shape}")
        batch_size, seq_len = x.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        # (B, seq_len) -> (B, seq_len, dim)
        h = tok_embeddings(x)
        print(f"tok embeddings {h.shape}")

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_comp = freqs_complex[start_pos:start_pos+seq_len]

        # consecutively apply all encoder blocks
        for layer in layers:
            h = layer(h, start_pos, freqs_comp)
        
        h = norm(h)
        op = output(h).astype(jnp.float32)
        return op




    
    