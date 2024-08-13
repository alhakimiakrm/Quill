import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import PoemDataset
from src.model import LSTMModel

def train_model(preprocessor, num_epochs=40, sequence_length=100, batch_size=256, learning_rate=0.001, progress_callback=None):
    # Preprocess the data
    sequences = preprocessor.preprocess()

    # Define model parameters
    vocab_size = len(preprocessor.vocab)
    embed_size = 512
    hidden_size = 1024
    num_layers = 3

    # Initialize the LSTM model
    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = PoemDataset(sequences, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    total_steps = len(dataloader)
    for epoch in range(num_epochs):
        model.train()
        
        for i, (inputs, targets) in enumerate(dataloader):
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

            # Send progress updates
            if progress_callback:
                progress_callback(epoch, num_epochs, i, total_steps, loss.item())

    return model, preprocessor

