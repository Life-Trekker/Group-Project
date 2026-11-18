import os
import torch
import matplotlib.pyplot as plt
from local_code.stage_4_code.Method_RNN_Generation import (
    TextGenData, Method_RNN_Generation, detokenize)

torch.manual_seed(2)
RNN_TYPE = 'GRU'
EPOCHS = 30

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'stage_4_data', 'text_generation', 'data')
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result', 'graphs')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result', 'result_data')
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('************ Train ************')
data = TextGenData(DATA_PATH, seq_len=20, min_freq=2)
model = Method_RNN_Generation(len(data.itos), embed_dim=128, hidden_dim=256,
                              num_layers=2, rnn_type=RNN_TYPE, dropout=0.4)
h = model.fit(data, epochs=EPOCHS, device=device)

print('************ Learning curves ************')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(h['epoch'], h['loss'], marker='o', label='Train')
a1.plot(h['epoch'], h['val_loss'], marker='s', label='Validation')
a1.set_xlabel('Epoch'); a1.set_ylabel('Cross-entropy loss')
a1.set_title('Loss per Epoch'); a1.legend(); a1.grid(True)
a2.plot(h['epoch'], h['val_ppl'], marker='o')
a2.set_xlabel('Epoch'); a2.set_ylabel('Validation perplexity')
a2.set_title('Validation Perplexity'); a2.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f'generation_curves_{RNN_TYPE}.png'), dpi=150)

torch.save({'state': model.state_dict(), 'itos': data.itos},
           os.path.join(RESULT_DIR, f'generation_model_{RNN_TYPE}.pt'))

print('************ Sample stories ************')
samples = []
for seed in (['what', 'did', 'the'], ['why', 'did', 'the'],
             ['what', 'do', 'you'], ['why', 'are', 'there']):
    for _ in range(3):
        text = detokenize(model.generate(data, seed, device=device))
        print(text)
        samples.append(text)
with open(os.path.join(RESULT_DIR, f'generation_samples_{RNN_TYPE}.txt'), 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(samples))

print('\n************ Manual test ************')
print('Type three starting words (or press Enter to quit).')
while True:
    line = input('> ').strip()
    if not line:
        break
    seed = line.lower().split()
    if len(seed) != 3:
        print('Please enter exactly three words.')
        continue
    unknown = [w for w in seed if w not in data.stoi]
    if unknown:
        print('Not in vocabulary, try other words:', unknown)
        continue
    print(detokenize(model.generate(data, seed, device=device)))
print('************ Finish ************')