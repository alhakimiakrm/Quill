import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.pre_processor import PoemPreprocessor, load_corpus
from src.dataset import PoemDataset
from src.model import LSTMModel

def train_model(corpus_path, num_epochs=20, sequence_length=10, batch_size=32, learning_rate=0.001):
    corpus = load_corpus(corpus_path)
    preprocessor = PoemPreprocessor(corpus)
    sequences = preprocessor.preprocess()

    vocab_size = len(preprocessor.vocab)
    embed_size = 128
    hidden_size = 256
    num_layers = 2

    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = PoemDataset(sequences, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        
        for i, (inputs, targets) in enumerate(dataloader):
            # Get the actual batch size for the current iteration
            batch_size_actual = inputs.size(0)
            hidden = model.init_hidden(batch_size_actual)
            hidden = tuple([h.to(device) for h in hidden])
            
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([h.data for h in hidden])
            
            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    return model, preprocessor
