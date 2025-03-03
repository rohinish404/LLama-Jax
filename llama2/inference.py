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
        batch_size = 1
        seq_len = 1
        dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        rngs = {'params': jax.random.PRNGKey(0)}
        
        # Initialize the model and get a params template
        init_params = model.init(rngs, dummy_input, 0)
        
        # Replace the randomly initialized weights with the loaded weights
        # This may need adjustment based on your exact parameter structure
        params = jax_params  # You may need to restructure jax_params to match init_params
        
        
        return LLAMA(model=model, tokenizer=tokenizer, model_args=llama_config, params=params)

    # @staticmethod
    # def convert_weights(ckpts, params):
    #     """Helper function to convert PyTorch weights to JAX compatible format."""
    #     jax_weights = {
    #         'transformer': {
    #             'wte': {'embedding': np.concatenate([ckpt['tok_embeddings.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=1)},
    #             'ln_f': {'kernel': ckpts[0]['norm.weight'].type(torch.float32).numpy()},
    #             'h': {
    #                 '%d' % (layer): {
    #                     'attention': {
    #                         'wq': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wq.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=0).transpose()},
    #                         'wk': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wk.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=0).transpose()},
    #                         'wv': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wv.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=0).transpose()},
    #                         'wo': {'kernel': np.concatenate([ckpt[f'layers.{layer}.attention.wo.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=1).transpose()},
    #                     },
    #                     'feed_forward': {
    #                         'w1': {'kernel': np.concatenate([ckpt[f'layers.{layer}.feed_forward.w1.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=0).transpose()},
    #                         'w2': {'kernel': np.concatenate([ckpt[f'layers.{layer}.feed_forward.w2.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=1).transpose()},
    #                         'w3': {'kernel': np.concatenate([ckpt[f'layers.{layer}.feed_forward.w3.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=0).transpose()},
    #                     },
    #                     'attention_norm': {'kernel': ckpts[0][f'layers.{layer}.attention_norm.weight'].type(torch.float32).numpy()},
    #                     'ffn_norm': {'kernel': ckpts[0][f'layers.{layer}.ffn_norm.weight'].type(torch.float32).numpy()},
    #                 } for layer in range(params['n_layers'])
    #             },
    #         },
    #         'lm_head': {'kernel': np.concatenate([ckpt['output.weight'].type(torch.float32).numpy() for ckpt in ckpts], axis=0).transpose()},
    #     }
    #     return jax_weights

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
            logits = self.model.apply(self.params, tokens[:, :cur_pos], cur_pos)
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
        """Sample from top-p (nucleus) distribution."""
        probs_sort, probs_idx = jax.lax.sort_key_val(
            probs, jnp.arange(probs.shape[-1], dtype=jnp.int32), 
            dimension=-1, is_stable=True, is_ascending=False
        )
        
        # Compute cumulative sum
        probs_sum = jnp.cumsum(probs_sort, axis=-1)
        
        # Create mask for values to keep (cumulative sum - current prob <= p)
        mask = probs_sum <= p
        
        # Zero out probabilities not selected by top-p
        probs_sort = jnp.where(mask, probs_sort, 0.0)
        
        # Renormalize the remaining probabilities
        probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
        
        # Sample a token index from the top-p distribution
        next_token_idx = jax.random.choice(
            key, 
            jnp.arange(probs.shape[-1]), 
            shape=(probs.shape[0],), 
            replace=True, 
            p=probs_sort
        )
        
        # Convert the sampled index back to the original vocabulary indices
        next_token = jnp.take_along_axis(probs_idx, next_token_idx[:, None], axis=-1)
        
        return next_token.squeeze(-1)

if __name__ == '__main__':
    
    
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
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