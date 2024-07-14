import torch
from torch.utils.data import Dataset

'''
This is a custom DataSet class for handling text sequences.
The class intializes with the sequences and sequence length.
__len__ returns the length of the dataset
__getitem__ retrieves a sequence and its corresponding target sequence at the given index.  
'''

class PoemDataset(Dataset):
    def __init__(self, sequences, sequence_length):
        self.sequences = sequences
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sequences) - self.sequence_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx:idx + self.sequence_length]),
            torch.tensor(self.sequences[idx + 1:idx + self.sequence_length + 1])
        )
