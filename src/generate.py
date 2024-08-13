import torch
import torch.nn.functional as F
from src.model import LSTMModel
import numpy as np

def generate_text(model, start_text, preprocessor, num_words, sequence_length):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    words = start_text.split()
    state_h, state_c = model.init_hidden(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    poem_lines = []
    current_line = []

    # Add the initial input to the current line
    current_line.extend(words)

    last_word = words[-1] if words else None

    for _ in range(num_words):
        input_indices = [
            preprocessor.word_to_idx.get(w, preprocessor.word_to_idx[preprocessor.unk_token])
            for w in words[-sequence_length:]
        ]
        x = torch.tensor([input_indices], dtype=torch.long).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[-1]
        p = F.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        
        # Avoid repeating the same word as the last one
        if last_word is not None:
            p[preprocessor.word_to_idx.get(last_word, 0)] *= 0.1
        
        # Normalize probabilities to sum to 1
        p = p / p.sum()

        word_idx = np.random.choice(len(p), p=p)
        next_word = preprocessor.idx_to_word[word_idx]
        last_word = next_word

        if next_word in ['.', '!', '?', ',', ';', ':', "'", '"', '(', ')', '[', ']', '{', '}', '-']:
            if current_line:
                current_line[-1] += next_word  # Attach punctuation to the last word
            else:
                current_line.append(next_word)  # If current_line is empty, start with the punctuation
        else:
            current_line.append(next_word)
            if len(current_line) > 7:  # Limit line length for better formatting
                poem_lines.append(' '.join(current_line))
                current_line = []


        words.append(next_word)

    if current_line:
        poem_lines.append(' '.join(current_line))  # Add the last line if not empty

    return '\n'.join(poem_lines)  # Join lines with newlines to form stanzas
