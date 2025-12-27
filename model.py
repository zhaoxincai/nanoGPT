# part 1: 导入相关的 package
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader


import math

torch.manual_seed(1024)


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)
        self.head_size = config.head_size
        self.register_buffer(
            'attention_mask',
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            )
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float('-inf')
        )
        weight = F.softmax(weight, dim = -1) / math.sqrt(self.head_size)

        weight = self.dropout(weight)
        output = weight @ v
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_head)
            ]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4* config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class EmbeddingWithPosition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        position_idx = torch.arange(0, config.seq_max_len, dtype=torch.float32).unsqueeze(-1)
        position_emb_fill = position_idx * torch.exp(-torch.arange(0, config.n_embd, 2) * math.log(10000.0)/ config.n_embd)
        position_encoding = torch.zeros(config.seq_max_len, config.n_embd)
        position_encoding[:, 0::2] = torch.sin(position_emb_fill)
        position_encoding[:, 1::2] = torch.cos(position_emb_fill)
        self.register_buffer('pos_encoding', position_encoding)
    def forward(self, x):
        x = self.token_embedding(x)
        x = x + self.pos_encoding.unsqueeze(0)[:,:x.size()[1], :]
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.emb=EmbeddingWithPosition(config)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.config = config
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean= 0.0, std= 0.02)

    def generate(self, idx):    ##  max_new_tokens 允许最大输出长度
        # is_question = True
        for _ in range(self.config.max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.seq_max_len else idx[:, -self.config.seq_max_len:]   ## 输入截断，输入【问题+当前回答】
            # print(idx_cond.shape)
            logits, _ = self.forward(idx_cond)   ## 输入进model
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)  ## 取概率最大的下一个数
            # print(idx_next)
            idx = torch.cat([idx, idx_next], dim=1)   ## 问题 回答拼接
            if idx_next == 50256:   ## 如果输出的是eos_token则直接停止
                break
            if idx.size(1) > self.config.seq_max_len:
                print("超出最大长度，生成结束")
                break
        return idx
    
    def forward(self, x, targets = None):
        batch_size, seq_len = x.size()
        x = self.emb(x)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            batch, seq_len, vocab_size =logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch* seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

if __name__=='__main__':
    from tokenizer import BPETokenizer
    tokenizer=BPETokenizer()
    tokenizer.load('tokenizer.bin')
    
    x=torch.randint(0,tokenizer.vocab_size(),(5,30))
    padding=torch.zeros(5,30)
    
    from config import GPTConfig
    gpt=GPT(GPTConfig)
    y=gpt(x,padding)
    print(y.shape)
