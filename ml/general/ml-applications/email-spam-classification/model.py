from torch import optim
from torch.nn import Linear
from transformers import BertTokenizer, BertModel, TrainingArguments
import lightning as L
from torch.nn.functional import cross_entropy
import torch

device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
class EmailClassifier(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        training_args = TrainingArguments(use_mps_device=True, output_dir="test_trainer")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', 
                                                                  config=config).to(device)
        self.fc = Linear(in_features=self.bert.config.hidden_size, out_features=2, device=device)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        x_tokenized = self.tokenizer(x, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
        x_tokenized_input_ids = x_tokenized['input_ids'].to(device)
        x_tokenized_attention_mask = x_tokenized['attention_mask'].to(device)
        z = self.bert(input_ids=x_tokenized_input_ids,
                          attention_mask=x_tokenized_attention_mask)
        z = z.pooler_output
        y_hat = self.fc(z)
        loss = cross_entropy(input=y_hat, target=y)
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer