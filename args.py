import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--d_model', default=768, help='模型的维度，即每个token的维度')
parser.add_argument('--d_ffn', default=768 * 4, help='FFN 的维度， 先将dim扩大四倍，再变回')
parser.add_argument('--n_head', default=12, help='多头自注意力的头数')
parser.add_argument('--size_per_head', default=64, help='每个头的维度')
parser.add_argument('--n_layer', default=6, help='编码器和解码器堆叠层数')
parser.add_argument('--ffn_drop_prob', default=0.1, help='ffn的dropout概率')
parser.add_argument('--att_drop_prob', default=0.1, help='self_attention的dropout概率')
parser.add_argument('--pe_len', default=512, help='self_attention的dropout概率')

parser.add_argument('--n_epoch', default=50, help='训练的轮数')
parser.add_argument('--lr', default=0.0001, help='学习率')
parser.add_argument('--checkpoint_save_path', default='./', help='ckpt保存')

# parser.add_argument('--src_vocab_size', default=0, help='encoder输入数据的词表大小')
# parser.add_argument('--tag_vocab_size', default=0, help='decoder输出数据的词表大小')
# parser.add_argument('--src_len', default=0, help='encoder输入的最大长度')
# parser.add_argument('--tag_len', default=0, help='decoder输入和输出的最大长度')


model_args = parser.parse_args()
