from data import EmailDataset, DataLoader
from torchvision.transforms import ToTensor, Compose
from model import EmailClassifier
import lightning as L
from transformers import BertConfig

def main():
    email_dataset = EmailDataset(emails_file='combined_data.csv')
    email_dataloader = DataLoader(dataset=email_dataset, batch_size=4)
    model_config = BertConfig()
    model = EmailClassifier(config=model_config)
    trainer = L.Trainer(limit_train_batches=500, max_epochs=16)
    trainer.fit(model=model, train_dataloaders=email_dataloader)

if __name__ == "__main__":
    main()