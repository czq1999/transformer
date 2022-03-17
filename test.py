# Test
import torch.cuda


from dataset import loader, tgt_vocab, idx2word
from model import greedy_decoder, Transformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('./runs_best.pth')
model_state_dict = checkpoint['state_dict']
args = checkpoint['args']
model = Transformer(args)

model.load_state_dict(model_state_dict)
model.to(device)

enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs.to(device)
for i in range(len(enc_inputs)):
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt_vocab["S"])
    predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
