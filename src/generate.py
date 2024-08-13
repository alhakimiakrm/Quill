import torch
import torch.nn.functional as F
import numpy as np

def generate_text(model, start_text, preprocessor, num_words, sequence_length, line_length=15, stanza_length=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    words = start_text.capitalize().split()
    state_h, state_c = model.init_hidden(1)
    state_h, state_c = state_h.to(device), state_c.to(device)

    poem_lines = []
    current_line = []
    stanza = []
    sentence_ended = False
    open_parentheses = 0

    last_word = words[-1] if words else None

    for _ in range(num_words):
        input_indices = [
            preprocessor.word_to_idx.get(w.lower(), preprocessor.word_to_idx[preprocessor.unk_token])
            for w in words[-sequence_length:]
        ]
        x = torch.tensor([input_indices], dtype=torch.long).to(device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        last_word_logits = y_pred[-1]
        p = F.softmax(last_word_logits, dim=0).detach().cpu().numpy()

        # Avoid repeating the same word as the last one
        if last_word is not None:
            p[preprocessor.word_to_idx.get(last_word.lower(), 0)] *= 0.1

        # Normalize probabilities to sum to 1
        p = p / p.sum()

        word_idx = np.random.choice(len(p), p=p)
        next_word = preprocessor.idx_to_word[word_idx]
        last_word = next_word

        # Handle punctuation, parentheses, apostrophes, and capitalization
        if next_word in ['.', '!', '?']:
            if current_line:
                current_line[-1] += next_word
                if open_parentheses == 0:  # Only end a sentence if no open parentheses
                    poem_lines.append(' '.join(current_line))
                    current_line = []
                    sentence_ended = True
        elif next_word == ',':
            if current_line:
                current_line[-1] += next_word
        elif next_word in [';', ':']:
            if current_line:
                current_line[-1] += next_word
        elif next_word == '(':
            current_line.append(next_word)
            open_parentheses += 1
        elif next_word == ')':
            if open_parentheses > 0:
                if current_line:
                    current_line[-1] += next_word
                open_parentheses -= 1
        elif next_word == "'":
            if current_line:
                current_line[-1] += next_word  # Attach apostrophe directly to the last word
        else:
            if sentence_ended:
                next_word = next_word.capitalize()
                sentence_ended = False
            else:
                next_word = next_word.lower()

            current_line.append(next_word)
            if len(current_line) >= line_length:  # Break line at specified length
                poem_lines.append(' '.join(current_line))
                current_line = []

            if len(poem_lines) >= stanza_length:  # Break stanza at specified line count
                stanza.append('\n'.join(poem_lines))
                poem_lines = []

        words.append(next_word)

    # Close any open parentheses before ending
    while open_parentheses > 0:
        current_line[-1] += ')'
        open_parentheses -= 1

    if current_line:
        poem_lines.append(' '.join(current_line))

    if poem_lines:
        stanza.append('\n'.join(poem_lines))  # Add the final stanza if not empty

    return '\n\n'.join(stanza)  # Separate stanzas with double newline for readability
