import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import sentences, make_data, MyDataSet, src_vocab, tgt_vocab, loader
from args import model_args
from utils import get_att_pad_mask, get_sinusoid_encoding_table, get_att_autoregressive_mask


# 缩放点积注意力,之所以把qkv抽取出来写，是让这个函数，不仅可以进行自注意力机制，也能在解码器端进行交叉注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.att_drop = nn.Dropout(self.args.att_drop_prob)

    def forward(self, q, k, v, att_mask):
        # q,k,v [B, n_head, N(seq_len), size_per_head]
        # att_score [B, n_head, N(seq_len), N(seq_len)]
        att_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.args.d_model)
        att_score.masked_fill_(att_mask, -1e9)
        att_score = nn.Softmax(dim=-1)(att_score)
        att_score = self.att_drop(att_score)

        context = torch.matmul(att_score, v)

        return context, att_score


# 多头自注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Wq = nn.Linear(self.args.d_model, self.args.n_head * self.args.size_per_head)
        self.Wk = nn.Linear(self.args.d_model, self.args.n_head * self.args.size_per_head)
        self.Wv = nn.Linear(self.args.d_model, self.args.n_head * self.args.size_per_head)
        self.fc = nn.Linear(self.args.n_head * self.args.size_per_head, self.args.d_model)  # 防止多头映射回去维度不一致的情况

        self.dot_product_attention = ScaledDotProductAttention(self.args)
        self.layer_norm = nn.LayerNorm(self.args.d_model)

    def forward(self, q, k, v, att_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        对于输入的qkv,需要按头进行维度变换。对于att_mask也需要按头进行维度变换
        '''

        # 转换输入 [B, N , D] -> [B, N, n_head , size_per_head] ->[B ,n_head, N, size_per_head]
        q_size = [q.size(0), q.size(1), self.args.n_head, self.args.size_per_head]
        k_size = [k.size(0), k.size(1), self.args.n_head, self.args.size_per_head]
        v_size = [v.size(0), v.size(1), self.args.n_head, self.args.size_per_head]

        # 计算QKV，并进行维度变换
        Q = self.Wq(q).view(*q_size).permute(0, 2, 1, 3)
        K = self.Wk(k).view(*k_size).permute(0, 2, 1, 3)
        V = self.Wv(v).view(*v_size).permute(0, 2, 1, 3)

        # 将att_mask 的第二个维度，扩充一下，扩充的倍数是n_head
        att_mask = att_mask.unsqueeze(1).repeat(1, self.args.n_head, 1, 1)

        context, attention = self.dot_product_attention(Q, K, V, att_mask)

        # 原本的多头维度再变换回去, permute是维度任意变换， transpose是交换某两个维度.
        # permute如果想view,需要保证内存空间是连续的。需要调用contiguous().或者使用reshape
        # [B, n_head, N, size_per_head]->[B,N,n_head,size_per_head]->[B, N, n_head*n_size_per_head] ->[B, N, d_model]
        context = context.transpose(1, 2).contiguous().view(q.size(0), q.size(1), -1)
        # 再使用fc确保，进行多头注意力前后的维度是一样的
        context = self.fc(context)
        return self.layer_norm(q + context), attention


# 前馈神经网络
class FFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ffn = nn.Sequential(nn.Linear(self.args.d_model, self.args.d_ffn),
                                 nn.GELU(),
                                 nn.Linear(self.args.d_ffn, self.args.d_model),
                                 nn.Dropout(self.args.ffn_drop_prob)
                                 )
        self.layer_norm = nn.LayerNorm(self.args.d_model)

    def forward(self, x):
        output = self.ffn(x)
        return self.layer_norm(x + output)


# 编码器一个block
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.SA = MultiHeadAttention(self.args)
        self.FFN = FFN(self.args)

    def forward(self, x, att_mask):
        # 输入的维度 [B, N , D]
        x, enc_att = self.SA(x, x, x, att_mask)
        x = self.FFN(x)
        return x, enc_att


# 编码器堆叠
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 词嵌入
        self.src_emb = nn.Embedding(self.args.src_vocab_size, self.args.d_model)
        # 把位置嵌入也做成词表
        self.sinusoid_table = get_sinusoid_encoding_table(self.args.pe_len + 1, self.args.d_model)
        self.pos_emb = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.args) for _ in range(self.args.n_layer)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        # 按token的index从pos_emb词表中读取位置embedding
        enc_outputs_index = torch.arange(0, enc_inputs.size(1)).to(enc_inputs.device).unsqueeze(0).repeat(
            enc_inputs.size(0), 1).to(torch.int64)
        pe = self.pos_emb(enc_outputs_index)
        enc_outputs = enc_outputs + pe
        enc_self_att_mask = get_att_pad_mask(enc_inputs, enc_inputs)
        enc_att_list = []
        for layer in self.encoder_layers:
            enc_outputs, enc_att = layer(enc_outputs, enc_self_att_mask)
            enc_att_list.append(enc_att)
        return enc_outputs, enc_att_list


# 解码器一个block
class DecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dec_self_att = MultiHeadAttention(self.args)
        self.dec_enc_att = MultiHeadAttention(self.args)
        self.ffn = FFN(self.args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''

        :param dec_inputs: 解码器输入
        :param enc_outputs: 编码器输出-交叉那部分
        :param dec_self_attn_mask: 解码器SA的mask
        :param dec_enc_attn_mask: 解码器和编码器交叉注意力的mask
        :return: 解码输出，SA的注意力权重，CA的注意力权重
        '''
        # MultiHeadAttention (q,k,v,mask[不是多头版的，在ScaledDotProductAttention中repeat为多头])
        dec_outputs, dec_self_att = self.dec_self_att(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_att = self.dec_enc_att(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        output = self.ffn(dec_outputs)

        return output, dec_self_att, dec_enc_att


# 解码器堆叠
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # 解码器段的词嵌入
        self.tgt_emb = nn.Embedding(self.args.tag_vocab_size, self.args.d_model)
        # 把位置嵌入也做成词表
        self.sinusoid_table = get_sinusoid_encoding_table(self.args.pe_len + 1, self.args.d_model)
        self.pos_emb = nn.Embedding.from_pretrained(self.sinusoid_table, freeze=True)

        self.decoder_layers = nn.ModuleList([DecoderLayer(self.args) for _ in range(self.args.n_layer)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        '''
        Transformer的解码器
        :param dec_inputs: decoder的输入，[batch_size, tgt_len]
        :param enc_inputs: encoder的输入，[batch_size, src_len]
        :param enc_outputs: encoder的输出，[batch_size, src_len, n_model]
        :return: output, dec_self_att, dec_enc_att
        '''
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs_index = torch.arange(0, dec_inputs.size(1)).to(dec_inputs.device).unsqueeze(0).repeat(
            dec_inputs.size(0), 1).to(torch.int64)
        pe = self.pos_emb(dec_outputs_index)
        dec_outputs = dec_outputs + pe

        # 解码器自注意力的mask矩阵
        dec_self_att_mask = get_att_pad_mask(dec_inputs, dec_inputs)
        dec_self_att_ar_mask = get_att_autoregressive_mask(dec_inputs).to(dec_inputs.device)
        # 判断mask矩阵相加 是否比0大，比0大的返回True。不看后面的+不看padding
        dec_self_att_mask = torch.gt((dec_self_att_ar_mask + dec_self_att_mask), 0)
        # print(dec_self_att_ar_mask == dec_self_att_mask)

        # 解码器和编码器结合那部分的mask矩阵
        dec_enc_att_mask = get_att_pad_mask(dec_inputs, enc_inputs)

        dec_self_att_list, dec_enc_att_list = [], []
        for layer in self.decoder_layers:
            dec_outputs, dec_self_att, dec_enc_att = layer(dec_outputs, enc_outputs, dec_self_att_mask,
                                                           dec_enc_att_mask)
            dec_self_att_list.append(dec_self_att)
            dec_enc_att_list.append(dec_enc_att)

        return dec_outputs, dec_self_att_list, dec_enc_att_list


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(self.args)
        self.decoder = Decoder(self.args)
        self.projection = nn.Linear(self.args.d_model, self.args.tag_vocab_size)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_att = self.encoder(enc_inputs)
        output, dec_self_att, dec_enc_att = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # [batch_size, tgt_len, d_model]-> [batch_size, tgt_len, tgt_vocab_size]
        logits = self.projection(output)
        return logits.view(-1, logits.size(-1)), enc_att, dec_self_att, dec_enc_att


def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    while not terminal:
        dec_input = torch.cat(
            [dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(dec_input.device)], -1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        print(next_word)
    return dec_input


if __name__ == "__main__":
    # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 读取数据

    train_loader = loader
    # 读取参数,并根据读入的数据，将args中的某些默认参数更新
    args = model_args
    args.src_vocab_size = len(src_vocab)
    args.tag_vocab_size = len(tgt_vocab)
    args.src_len = len(sentences[0][0].split(" "))
    args.tag_len = len(sentences[0][1].split(" "))
    print(args)

    # 创建模型，构建优化器， 选择损失函数
    model = Transformer(args).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for idx, (enc_inputs, dec_inputs, dec_labels) in enumerate(train_loader):
        enc_inputs, dec_inputs, dec_labels = enc_inputs.to(device), dec_inputs.to(device), dec_labels.to(device)
        output, enc_att, dec_self_att, dec_enc_att = model(enc_inputs, dec_inputs)
        loss = criterion(output, dec_labels.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
