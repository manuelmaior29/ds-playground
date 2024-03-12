from torch.utils.data import Dataset, DataLoader
import pandas as pd

class EmailDataset(Dataset):
    def __init__(self, emails_file, transform=None, target_transform=None) -> None:
        self.data = pd.read_csv(emails_file).to_numpy()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        email = self.data[idx][1]
        target = self.data[idx][0]
        if self.transform:
            email = self.transform(email)
        if self.target_transform:
            target = self.target_transform(target)
        return email, target