from model import Transformer, ModelArgs
from sentencepiece import SentencePieceProcessor
import time
import torch
from pathlib import Path
import json
import numpy as np
from flax.core.frozen_dict import unfreeze, freeze
import jax
import jax.numpy as jnp
from typing import Optional
from tqdm import tqdm

def replace_params(init_params, loaded_params):
    """
    Maps loaded parameters to the expected model structure.
    
    Args:
        init_params: The initialized parameters from model.init() with the expected structure
        loaded_params: The loaded parameters from checkpoint files
    
    Returns:
        Properly structured parameters that match the model's expectations
    """
    import jax
    import jax.numpy as jnp
    from flax.core.frozen_dict import freeze, unfreeze
    
    # Convert to mutable dictionaries for easier manipulation
    init_params_dict = unfreeze(init_params)
    loaded_params_dict = unfreeze(loaded_params)
    
    # Create a mapping from loaded params to init params structure
    param_mapping = {
        # Embedding layer
        'transformer.wte.embedding': ('params', 'Embed_0', 'embedding'),
        
        # Output layer
        'lm_head.kernel': ('params', 'Dense_0', 'kernel'),
        
        # Final layer norm
        'transformer.ln_f.kernel': ('params', 'LayerNorm_0', 'scale'),
    }
    
    # Map transformer layers
    for i in range(32):  # Assuming 32 encoder blocks based on the provided structure
        # Attention layers
        param_mapping[f'transformer.h.{i}.attention.wq.kernel'] = ('params', f'EncoderBlock_{i}', 'SelfAttention_0', 'Dense_0', 'kernel')
        param_mapping[f'transformer.h.{i}.attention.wk.kernel'] = ('params', f'EncoderBlock_{i}', 'SelfAttention_0', 'Dense_1', 'kernel')
        param_mapping[f'transformer.h.{i}.attention.wv.kernel'] = ('params', f'EncoderBlock_{i}', 'SelfAttention_0', 'Dense_2', 'kernel')
        param_mapping[f'transformer.h.{i}.attention.wo.kernel'] = ('params', f'EncoderBlock_{i}', 'SelfAttention_0', 'Dense_3', 'kernel')
        
        # Feed-forward layers
        param_mapping[f'transformer.h.{i}.feed_forward.w1.kernel'] = ('params', f'EncoderBlock_{i}', 'FeedForward_0', 'Dense_0', 'kernel')
        param_mapping[f'transformer.h.{i}.feed_forward.w2.kernel'] = ('params', f'EncoderBlock_{i}', 'FeedForward_0', 'Dense_1', 'kernel')
        param_mapping[f'transformer.h.{i}.feed_forward.w3.kernel'] = ('params', f'EncoderBlock_{i}', 'FeedForward_0', 'Dense_2', 'kernel')
        
        # Layer norms
        param_mapping[f'transformer.h.{i}.attention_norm.kernel'] = ('params', f'EncoderBlock_{i}', 'LayerNorm_0', 'scale')
        param_mapping[f'transformer.h.{i}.ffn_norm.kernel'] = ('params', f'EncoderBlock_{i}', 'LayerNorm_1', 'scale')
    
    # Function to get value from nested dictionaries using path
    def get_nested(d, path):
        for key in path:
            if key in d:
                d = d[key]
            else:
                return None
        return d
    
    # Function to set value in nested dictionaries using path
    def set_nested(d, path, value):
        for key in path[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[path[-1]] = value
    
    # Apply the mapping
    mapped_params = {}
    for src_path_str, dest_path in param_mapping.items():
        # Parse the source path
        src_path = src_path_str.split('.')
        
        # Get the parameter from loaded_params
        param_value = get_nested(loaded_params_dict, src_path)
        
        if param_value is not None:
            # Set the parameter in the init_params structure
            set_nested(mapped_params, dest_path, param_value)
    
    # Convert back to frozen dict
    return freeze(mapped_params)

class LLAMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs, params):
        self.model = model
        self.tokenizer = tokenizer
        self.model_args = model_args
        self.params = params
    
    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, verbose:bool):
        prev_time = time.time()

        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0, "No checkpoints files found"
            ckpts = {}
            for i, chk_path in enumerate(checkpoints):
                if verbose:
                    print(f"Loading checkpoint {i+1} of {len(checkpoints)} ...")
                checkpoint = torch.load(chk_path, map_location="cpu")
                if verbose:
                    print('Loaded.')
                ckpts[int(chk_path.name.split('.', maxsplit=2)[1])] = checkpoint
                del checkpoint
            ckpts = [ckpts[i] for i in sorted(list(ckpts.keys()))]
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f}s')
            prev_time = time.time()
            
            for ckpt in ckpts:
                for name in ckpt.keys():
                    print(name)

        
        with open(Path(checkpoint_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        jax_weights = {
            'transformer': {
                'wte': {'embedding': np.concatenate([ckpt['tok_embeddings.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=1)},
                'ln_f': {'kernel': ckpts[0]['norm.weight'].type(torch.float16).numpy()},
                'h': {
                    '%d' % (layer): {
                        'attention': {
                            'wq': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wq.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()},
                            'wk': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wk.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()},
                            'wv': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wv.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()},
                            'wo': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wo.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=1).transpose()},
                        },
                        'feed_forward': {
                            'w1': {'kernel': np.concatenate([ckpt[f'layers.{layer}.feed_forward.w1.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()},
                            'w2': {'kernel': np.concatenate([ckpt[f'layers.{layer}.feed_forward.w2.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=1).transpose()},
                            'w3': {'kernel': np.concatenate([ckpt[f'layers.{layer}.feed_forward.w3.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()},
                        },
                        'attention_norm': {'kernel': ckpts[0][f'layers.{layer}.attention_norm.weight'].type(torch.float16).numpy()},
                        'ffn_norm': {'kernel': ckpts[0][f'layers.{layer}.ffn_norm.weight'].type(torch.float16).numpy()},
                    } for layer in range(params['n_layers'])
                },
            },
            'lm_head': {'kernel': np.concatenate([ckpt['output.weight'].type(torch.float16).numpy() for ckpt in ckpts], axis=0).transpose()},
        }

        # Load tokenizer
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        
        # Update parameters
        params.update({'vocab_size': tokenizer.vocab_size(), 'max_seq_len': max_seq_len})
        llama_config = LLAMA.config_from_params(ModelArgs(**params))

        # Convert parameters to JAX format and freeze
        jax_params = freeze(jax.tree.map(lambda x: jnp.asarray(x), jax_weights))

        # Initialize the model
        model = Transformer(llama_config)   

        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
        init_params = model.init(key, dummy_input, 0)
        
        # Print parameter structure to understand the model's expectations
        if verbose:
            print("Model expects parameters with this structure:")
            print(jax.tree_map(lambda x: x.shape if hasattr(x, 'shape') else None, init_params))   
        mapped_params = replace_params(init_params, jax_params)  
            
        return LLAMA(model=model, tokenizer=tokenizer, model_args=llama_config, params=mapped_params)


    @staticmethod
    def config_from_params(args: ModelArgs) -> ModelArgs:
        """Converts ModelArgs for initializing the model configuration."""
        return args
    
    def text_completion(self, prompts: list[str], temperature: float= 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.model_args.max_seq_len - 1
        
        from tqdm import tqdm  # Add missing import
        
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        
        batch_size = len(prompt_tokens)
        assert batch_size <= self.model_args.max_batch_size, f"batch size must be less than or equal to {self.model_args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.model_args.max_seq_len, f"prompt length must be less than or equal to {self.model_args.max_seq_len}"
        total_len = min(self.model_args.max_seq_len, max_gen_len + max_prompt_len)
        
        pad_id = self.tokenizer.pad_id()
        tokens = jnp.full((batch_size, total_len), pad_id, dtype=jnp.int32)
        
        # Use JAX's functional update instead of in-place assignment
        for k, t in enumerate(prompt_tokens):
            tokens = tokens.at[k, :len(t)].set(jnp.array(t, dtype=jnp.int32))
        
        eos_reached = jnp.array([False] * batch_size)
        prompt_tokens_mask = tokens != pad_id
        
        # Initialize PRNG key properly
        key = jax.random.PRNGKey(int(time.time()))
        
        cur_iterator = tqdm(range(1, total_len), desc="Generating Tokens")
        
        for cur_pos in cur_iterator:
            print(f"tokens[:, :cur_pos] {tokens[:, :cur_pos]}")
            logits = self.model.apply(self.params, tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                probs = jax.nn.softmax(logits[:, -1] / temperature, axis=1)
                key, subkey = jax.random.split(key)  # Split key for randomness
                next_token = self.sample_top_p(probs, top_p, subkey)  # Pass the key
            else:
                next_token = jnp.argmax(logits[:, -1], axis=1)
            next_token = next_token.reshape(-1)
            
            # Apply prompt mask
            next_token = jnp.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            
            # Update tokens functionally
            tokens = tokens.at[:, cur_pos].set(next_token)
            
            # Update EOS tracking
            eos_reached = eos_reached | ((~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id()))
            if jnp.all(eos_reached):
                break
        
        # Convert to Python lists for output
        tokens_numpy = np.array(tokens)
        out_tokens = []
        out_text = []
        
        for prompt_index, current_prompt_tokens in enumerate(tokens_numpy.tolist()):
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        
        return (out_tokens, out_text)
    def sample_top_p(self, probs, p, key):
        # Create an index array that matches the batch dimension of probs
        batch_size = probs.shape[0]
        vocab_size = probs.shape[-1]
        
        # Create indices tensor with same batch dimension as probs
        indices = jnp.tile(jnp.arange(vocab_size, dtype=jnp.int32)[None, :], (batch_size, 1))
        
        probs_sort, probs_idx = jax.lax.sort_key_val(
            probs, indices, 
            dimension=-1, is_stable=True
        )
        print(f"probs sort {probs_sort.shape}")
        
        # Rest of your implementation remains the same
        probs_sum = jnp.cumsum(probs_sort, axis=-1)
        print(f"probs sum {probs_sum.shape}")
        mask = probs_sum <= p
        print(f"mask {mask.shape}")
        probs_sort = jnp.where(mask, probs_sort, 0.0)
        probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
        print(f"probs sort {probs_sort.shape}")
        
        def sample_fn(k, ps):
            return jax.random.choice(k, jnp.arange(vocab_size), shape=(), replace=True, p=ps)

        keys = jax.random.split(key, batch_size)
        next_token_idx = jax.vmap(sample_fn)(keys, probs_sort)

        next_token = jnp.take_along_axis(probs_idx, next_token_idx[:, None], axis=-1)
        return next_token.squeeze(-1)
if __name__ == '__main__':
    
    
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        # """Translate English to French:
        
        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
        # # Zero shot prompt
        # """Tell me if the following person is actually Doraemon disguised as human:
        # Name: Umar Jamil
        # Decision: 
        # """
    ]
    
    model = LLAMA.build(
        checkpoint_dir='llama-2-7b',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        verbose=True
    )
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)