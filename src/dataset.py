import torch
from torch.utils.data import Dataset

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
