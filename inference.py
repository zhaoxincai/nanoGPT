from model import GPT
from dataset import Return_Question_tensor
from config import *
import torch 
from tokenizer import BPETokenizer
import torch.nn.functional as F
import random
import tiktoken

enc = tiktoken.get_encoding("gpt2")

config = GPTConfig()
model = GPT(config)
model.to(config.device)

try:  
    checkpoint=torch.load('./checkpoints/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
except:
    pass

model.eval()

x = Return_Question_tensor("在查处虚开增值税专用发票案件中",config).to(config.device)
out = model.generate(x)
output = enc.decode(out[:,config.max_length:].data.to("cpu").numpy()[0])
print("output:",output)
