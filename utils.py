import numpy as np
import torch


# todo: 3.17写一下这个

# 这个attention_mask是简单的按padding的mask

def get_att_pad_mask(seq_q, seq_k):
    '''
    表示q句子中的每个单词，对k句子的每个单词是否可见，主要是padding mask
    :param seq_q:  [batch_size, seq_q_len]
    :param seq_k:  [batch_size, seq_k_len]
    :return:
    '''
    batch_size, seq_q_len = seq_q.size()
    batch_size, seq_k_len = seq_k.size()
    # seq_k句子进行mask， 然后再扩充一下维度，表示seq_q的每个单词对seq_k的padding mask
    # [batch_size, seq_k_len]->[batch_size, 1, seq_k_len]->[batch_size, seq_q_len,seq_k_len]
    mask = seq_k.data.eq(0)
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, seq_q_len, 1)  # 这里说一下expand和repeat ,repeat中参数为1表示不变，expand中参数为原始表示不变
    return mask


def get_att_autoregressive_mask(dec_seq):
    '''
    防止decoder看到未来的信息，
    :param dec_inputs: [batch_size,tgt_len]
    :return: mask [batch_size, tgt_len,tgt_len], 返回的应该是上三角矩阵
    '''
    mask_shape = [dec_seq.size(0), dec_seq.size(1), dec_seq.size(1)]
    numpy_mask = np.triu(np.ones(mask_shape), 1)
    mask = torch.from_numpy(numpy_mask).byte()
    return mask




def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


if __name__ == '__main__':
    seq_q = torch.Tensor([[1, 2, 3, 0, 0],
                          [1, 2, 3, 1, 0],
                          [1, 2, 3, 1, 1],
                          ])

    seq_k = torch.Tensor([[1, 2, 3, 0, 0, 0],
                          [1, 2, 3, 1, 0, 0],
                          [1, 2, 3, 1, 1, 1],
                          ])

    get_att_pad_mask(seq_q, seq_k)
