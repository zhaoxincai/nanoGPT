from collections import OrderedDict
import pickle
import re
from tqdm import tqdm

class BPETokenizer:
    def __init__(self):
        self.b2i = OrderedDict()
        self.i2d = OrderedDict()
        self.next_id = 0

        self.sp_s2i = {}
        self.sp_i2s = {}
    
    def _pair_stats(self, tokens, stats):
        for i in range(len(tokens) - 1):
            new_token = tokens[i] + tokens[i+1]
            if new_token not in stats:
                stats[new_token] = 0
            stats[new_token] += 1

    def _merge_pair(self, tokens, new_token):
        merged_token = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] + tokens[i+1] == new_token:
                merged_token.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                merged_token.append(tokens[i])
                i += 1
        return merged_token

    def train(self, text_list, vocab_size):
        for i in range(256):
            self.b2i[bytes([i])] = i
        self.next_id = 256

        tokens_list = []
        for text in text_list:
            tokens = [bytes([b]) for b in text.encode('utf-8')]
            tokens_list.append(tokens)
        
        progress = tqdm(total = vocab_size , initial = 256)

        while True:
            if self.next_id >= vocab_size:
                break
            stats = {}
            for tokens in tokens_list:
                self._pair_stats(tokens, stats)
            
            if not stats:
                break
            
            new_token = max(stats, key = stats.get)

            new_tokens_list = []
            for tokens in tokens_list:
                new_tokens_list.append(self._merge_pair(tokens, new_token))
            tokens_list = new_tokens_list

            self.b2i[new_token] = self.next_id
            self.next_id += 1

            progress.update(1)
        self.i2b = {v:k for k, v in self.b2i.items()}
    def vocab_size(self):
        return self.next_id

    def vocab(self):
        v={}
        v.update(self.i2b)
        v.update({id:token.encode('utf-8') for id,token in self.sp_i2s.items()})
        return v
    
    # 特殊token
    def add_special_tokens(self,special_tokens):
        for token in special_tokens:
            if token not in self.sp_s2i:
                self.sp_s2i[token]=self.next_id
                self.sp_i2s[self.next_id]=token
                self.next_id+=1
    
    def encode(self,text):
        # 特殊token分离
        pattern='('+'|'.join([re.escape(tok) for tok in self.sp_s2i])+')'
        splits=re.split(pattern,text) # [ '<|im_start|>', 'user', '<||>' ]
        
        # 编码结果
        enc_ids=[]
        enc_tokens=[]
        for sub_text in splits:
            if sub_text in self.sp_s2i: # 特殊token，直接对应id
                enc_ids.append(self.sp_s2i[sub_text])
                enc_tokens.append(sub_text.encode('utf-8'))
            else:
                tokens=[bytes([b]) for b in sub_text.encode('utf-8')]
                while True:
                    # 统计相邻token频率
                    stats={}
                    self._pair_stats(tokens,stats)
                    
                    # 选择合并后id最小的pair合并（也就是优先合并短的）
                    new_token=None
                    for merge_token in stats:
                        if merge_token in self.b2i and (new_token is None or self.b2i[merge_token]<self.b2i[new_token]):
                            new_token=merge_token
                    
                    # 没有可以合并的pair，退出
                    if new_token is None:
                        break

                    # 合并pair
                    tokens=self._merge_pair(tokens,new_token)
                enc_ids.extend([self.b2i[tok] for tok in tokens])
                enc_tokens.extend(tokens)
        return enc_ids,enc_tokens
    
    def decode(self,ids):
        bytes_list=[]
        for id in ids:
            if id in self.sp_i2s:
                bytes_list.append(self.sp_i2s[id].encode('utf-8'))
            else:
                bytes_list.append(self.i2b[id])
        return b''.join(bytes_list).decode('utf-8',errors='replace')
    
    def save(self,file):
        with open(file,'wb') as fp:
            fp.write(pickle.dumps((self.b2i,self.sp_s2i,self.next_id)))
    
    def load(self,file):
        with open(file,'rb') as fp:
            self.b2i,self.sp_s2i,self.next_id=pickle.loads(fp.read())
        self.i2b={v:k for k,v in self.b2i.items()}
        self.sp_i2s={v:k for k,v in self.sp_s2i.items()}

if __name__=='__main__':
    # 加载语料
    # cn=open('dataset/train-cn.txt','r').read()
    # en=open('dataset/train-en.txt','r').read()
    
    # # 训练
    # tokenizer=BPETokenizer()
    # tokenizer.train(text_list=[cn,en],vocab_size=300)
    
    # # 特殊token
    # tokenizer.add_special_tokens((['<|im_start|>','<|im_end|>','<|endoftext|>','<|padding|>']))
    
    # # 保存
    # tokenizer.save('tokenizer.bin')
    
    # 还原
    tokenizer=BPETokenizer()
    tokenizer.load('tokenizer.bin')
    print('vocab size:',tokenizer.vocab_size())
    
    # 编码
    ids,tokens=tokenizer.encode('<|im_start|>system\nyou are a helper assistant\n<|im_end|>\n<|im_start|>user\n今天的天气怎么样？？\n<|im_end|><|im_start|>assistant\n')
    print('encode:',ids,tokens)
    
    # 解码
    s=tokenizer.decode(ids)
    print('decode:','\n',s)
    
    # 打印词典
    print('vocab:',tokenizer.vocab())
