import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import PoemDataset
from src.model import LSTMModel

def train_model(preprocessor, num_epochs=40, sequence_length=100, batch_size=256, learning_rate=0.001):
    # Preprocess the data
    sequences = preprocessor.preprocess()

    # Define model parameters
    vocab_size = len(preprocessor.vocab)
    embed_size = 512  # Increased for better representation
    hidden_size = 1024  # Increased for deeper learning
    num_layers = 3  # Increased to add more depth to the LSTM

    # Initialize the LSTM model
    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    
    # Set device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Using AdamW for better weight decay handling

    # Create dataset and dataloader
    dataset = PoemDataset(sequences, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        for i, (inputs, targets) in enumerate(dataloader):
            # Move data to the GPU
            batch_size_actual = inputs.size(0)
            hidden = model.init_hidden(batch_size_actual)
            hidden = tuple([h.to(device) for h in hidden])
            
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([h.data for h in hidden])
            
            # Forward pass
            model.zero_grad()
            output, hidden = model(inputs, hidden)
            
            # Compute loss
            loss = criterion(output, targets.view(-1))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Print loss every 100 steps
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        # Save the model checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    return model, preprocessor
