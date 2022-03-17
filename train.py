import torch
from torch import nn
from torch.optim import Adam, SGD

from args import model_args
from dataset import loader, src_vocab, tgt_vocab, sentences, idx2word
from model import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

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
optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.99)
criterion = nn.CrossEntropyLoss()

min_loss = 1e9
for epoch in range(args.n_epoch):
    for idx, (enc_inputs, dec_inputs, dec_labels) in enumerate(train_loader):
        enc_inputs, dec_inputs, dec_labels = enc_inputs.to(device), dec_inputs.to(device), dec_labels.to(device)
        output, enc_att, dec_self_att, dec_enc_att = model(enc_inputs, dec_inputs)
        loss = criterion(output, dec_labels.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss < min_loss:
            torch.save({"state_dict": model.state_dict(), 'args': args}, args.checkpoint_save_path + '/runs_best.pth')
