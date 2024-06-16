import logging
import torch
import torch.nn as nn
import os
import re
import jieba
from tqdm import trange
from gensim.models import Word2Vec
import numpy as np



#数据读取
#文件夹路径
file_path = r'D:\Deeplearning\鹿鼎记.txt'
file1 = open(file_path, 'r', encoding='utf-8')#打开当前txt文件、只读
text = file1.read()#读取当前文件的所有内容
file1.close()#关闭文件
text = text.replace("本书来自www.cr173.com免费txt小说下载站", "")
text = text.replace("更多更新免费电子书请关注www.cr173.com", "")
#数据预处理
#清除乱码
#判断char是否是乱码
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('，','。','：','？','“','”','！','；','、','《','》','——'):
            return True
    return False

# 将每行的list合成一个长字符串
data = ''.join(text)
data = [char for char in data if is_uchar(char)]
data = ''.join(data)

#模型搭建

class Net(nn.Module):
    def __init__(self, onehot_num):
        super(Net, self).__init__()   #继承父类的构造函数，重写时以子类函数为准
        onehot_size = onehot_num
        embedding_size = 256
        n_layer = 2
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)  # 编码
        self.encode = torch.nn.Sequential(nn.Linear(onehot_size, embedding_size), nn.Dropout(0.5), nn.ReLU())  # 解码
        self.decode = torch.nn.Sequential(nn.Linear(embedding_size, onehot_size), nn.Dropout(0.5), nn.Sigmoid())
                     #torch.nn.Sequential将模块组成流水线，输入依次通过这些模块得到输出。
    def forward(self, x):  # 输入
        em = self.encode(x).unsqueeze(dim=1)  # 输出
        out, (h, c) = self.lstm(em)
        res = 2 * (self.decode(out[:, 0, :]) - 0.5)
        return res

#模型训练
def train():
    embed_size = 1024
    epochs = 25
    end_num = 10
    #文本数据读取，进一步处理数据
    all_text = data
    text_terms = list()
    for text_line in all_text.split('。'):
        seg_list = list(jieba.cut(text_line, cut_all=False))  # 使用精确模式
        if len(seg_list) < 5:
            continue
        seg_list.append("END")
        text_terms.append(seg_list)

    # 训练word2vec模型，计算词向量

    if not os.path.exists('model.model'):
        model = Word2Vec(sentences=text_terms, sg=0, vector_size=embed_size, min_count=1, window=10, epochs=10)
        model.save('model.model')

    sequences = text_terms
    vec_model = Word2Vec.load('model.model')
    model = Net(embed_size)#.cuda()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    for epoch_id in range(epochs):
        for idx in trange(0, len(sequences) // end_num - 1):
            seq = []
            for k in range(end_num):
                seq += sequences[idx + k]
            target = []
            for k in range(end_num):
                target += sequences[idx + end_num + k]
            input_seq = torch.zeros(len(seq), embed_size)
            for k in range(len(seq)):
                input_seq[k] = torch.tensor(vec_model.wv[seq[k]])
                target_seq = torch.zeros(len(target), embed_size)
            for k in range(len(target)):
                target_seq[k] = torch.tensor(vec_model.wv[target[k]])
            all_seq = torch.cat((input_seq, target_seq), dim=0)
            optimizer.zero_grad()
            out_res = model(all_seq[:-1])#.cuda()
            f1 = ((out_res[-target_seq.shape[0]:] ** 2).sum(dim=1)) ** 0.5
            f2 = ((target_seq ** 2).sum(dim=1)) ** 0.5#.cuda()
            loss = (1 - (out_res[-target_seq.shape[0]:] * target_seq).sum(dim=1) / f1 / f2).mean()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:
                print("loss: ", loss.item(), " in epoch ", epoch_id, " res: ",
                      out_res[-target_seq.shape[0]:].max(dim=1).indices, target_seq.max(dim=1).indices)
        state = {"models": model.state_dict()}
        torch.save(state, "model.pth")

#测试函数
def Atest():
    embed_size = 1024
    text_data = "韦小宝本盼他一刀砍在自己胸口，自己有宝衣护身，不会丧命，便可将四名喇嘛吓得逃走，哪知他不砍胸而砍背，将自己推入他怀中，正好乘机用匕首戳他几剑，只是在对方胯下爬出，未名太过狼狈，临危逃命，也顾不得英雄还是狗熊了。"
    text = ""
    for line in text_data:
        line = re.sub('\s', '', line)
        line = re.sub('！', '。', line)
        line = re.sub('？', '。', line)
        line = re.sub('，', '。', line)
        line = re.sub('[\u0000-\u3001]', '', line)
        line = re.sub('[\u3003-\u4DFF]', '', line)
        line = re.sub('[\u9FA6-\uFFFF]', '', line)
        text += line
    text_terms = list()
    test_len = 0
    for text_line in text.split('。'):
        seg_list = list(jieba.cut(text_line, cut_all=False))  # 使用精确模式
        if len(seg_list) < 5:
            continue
        seg_list.append("END")
        test_len = test_len + len(seg_list)
        text_terms.append(seg_list)


    checkpoint = torch.load("model.pth")

    model = Net(embed_size).eval()
    model.load_state_dict(checkpoint["models"])
    vec_model = Word2Vec.load('model.model')

    seqs = []
    for sequence in text_terms:
        seqs += sequence

    input_seq = torch.zeros(len(seqs), embed_size)
    result = ""
    with torch.no_grad():
        for k in range(len(seqs)):
            input_seq[k] = torch.tensor(vec_model.wv[seqs[k]])
        end_num = 0
        length = 0
        while end_num < 10 and length < test_len:
            print("length: ", length)
            out_res = model(input_seq)[-2:-1]
            key_value = vec_model.wv.most_similar(positive=np.array(out_res.cpu()), topn=20)
            key = key_value[np.random.randint(20)][0]
            if key == "END":
                result += "。"
                end_num += 1
            else:
                result += key
            length += 1
            input_seq = torch.cat((input_seq, out_res), dim=0)
    print(result)

if __name__ == "__main__":
    train()
    Atest()