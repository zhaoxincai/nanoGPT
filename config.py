import torch
from dataclasses import dataclass

# special tokens
BOS='<|beginoftext|>'
EOS='<|endoftext|>'
PAD='<|padding|>'
IM_START='<|im_start|>'
IM_END='<|im_end|>'

VOCAB_SIZE=1024 


@dataclass
class GPTConfig:
    seq_max_len: int = 512
    max_new_tokens: int = 100

    block_size: int = 512
    batch_size: int = 16
    n_layer: int = 12
    n_head: int =12
    n_embd: int = 768
    hidden_dim: int = 768

    dropout: float = 0.1
    head_size: int = int(n_embd // n_head)
    vocab_size: int = 50257
    eos_token: int = 50256
    max_length: int = 200
    device: torch.device =  torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
