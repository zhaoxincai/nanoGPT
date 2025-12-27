import tiktoken
import json
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, path):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = 512

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special= {"<|endoftext|>"}
        )[0]
        self.encoded_data = []
        self.max_lines = 1000
        raw_data = []
        with open(path, "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > self.max_lines:
                    break
                try: 
                    text = json.loads(line.strip())
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
            full_encoded = []
            for text in raw_data:
                encoded_text = self.enc.encode(text)
                full_encoded.extend(encoded_text + [self.eos_token])
            
            for i in range(0, len(full_encoded), self.block_size):
                chunk = full_encoded[i:i + self.block_size + 1]
                if len(chunk) < self.block_size + 1:
                    chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
                self.encoded_data.append(chunk)

    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)

import numpy as np
from config import GPTConfig


# def Return_Question_tensor(question,config):   ## 对question进行序列化 ，如果输入过长可以增加max_length

#     enc = tiktoken.get_encoding("gpt2")
#     encoded_text = enc.encode(question)
#     full_encoded = encoded_text + [config.eos_token]
#     encoded_data = []
#     assert len(full_encoded) < config.max_length   ## embedding 之后的长度不能大于max_length

#     for i in range(0, 1, 100):
#         # 多取一个 Token 作为目标
#         chunk = full_encoded[i:i + config.max_length + 1]
#         # 如果长度不够，用 eos_token 填充
#         if len(chunk) < config.max_length + 1:
#             chunk = chunk + [config.eos_token] * (config.max_length  - len(chunk))
#         encoded_data.append(chunk)
#     return torch.tensor(encoded_data,dtype=torch.long)


def Return_Question_tensor(question, config):
    enc = tiktoken.get_encoding("gpt2")
    encoded_text = enc.encode(question)
    full_encoded = encoded_text + [config.eos_token]

    assert len(full_encoded) <= config.max_length, "输入太长"

    # pad 到 max_length
    pad_len = config.max_length - len(full_encoded)
    full_encoded = full_encoded + [config.eos_token] * pad_len

    # shape = [1, max_length]
    return torch.tensor([full_encoded], dtype=torch.long)

if __name__ == '__main__':
    enc = tiktoken.get_encoding("gpt2")
    s = Return_Question_tensor("你好,你好", config=GPTConfig())
    print(s.shape)
    print(enc.decode(s[0].tolist()))

# if __name__ == '__main__':
#     enc = tiktoken.get_encoding("gpt2")
#     s= Return_Question_tensor("你好,你好",config=GPTConfig())
#     print(np.array(s).shape)
#     print(enc.decode(np.array(s[1])))
    # train_dataset = MyDataset('./data/mobvoi_seq_monkey_general_open_corpus_1000.json')
    # print(len(train_dataset))
    # print(train_dataset[0])
