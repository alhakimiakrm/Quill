import torch
import torch.nn.functional as F
from src.model import LSTMModel
import numpy as np

'''
Implements LSTM (Long Short Term Memory) model for generating text from trained model.
The function below initializes the model's hidden state. It then iteratively predicts
the next word based on teh current sequence, appends the predicted word to the sequence and returns
the generated text. 
'''

def generate_text(model, start_text, preprocessor, num_words, sequence_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    words = start_text.split()
    state_h, state_c = model.init_hidden(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    for _ in range(num_words):
        x = torch.tensor([[preprocessor.word_to_idx[w] for w in words[-sequence_length:]]], dtype=torch.long).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[-1]  # Get logits for the last word
        p = F.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_idx = np.random.choice(len(p), p=p)
        words.append(preprocessor.idx_to_word[word_idx])

    return ' '.join(words)
