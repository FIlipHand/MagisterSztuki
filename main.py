import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel

import torch.nn.functional as F
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetTok, DataCollator
from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

VOCAB_SIZE = 5000

print('############### CREATING TOKENIZER ###############')
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
tokenizer = REMI(config)

# Trains the tokenizer with BPE, and save it to load it back later
midi_paths = list(Path("./data/maestro-v3.0.0").glob("**/*.midi"))
# tokenizer.learn_bpe(vocab_size=VOCAB_SIZE, files_paths=midi_paths)
tokenizer.save_params(Path("tokenizer.json"))

# Creates a Dataset and a collator to be used with a PyTorch DataLoader to train a model
dataset = DatasetTok(
    files_paths=midi_paths,
    min_seq_len=100,
    max_seq_len=1024,
    tokenizer=tokenizer,
)
collator = DataCollator(
    tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"]
)


print('############### CREATING MODEL ###############')

config = MambaConfig(d_model=384,
                     n_layer=4,
                     vocab_size=VOCAB_SIZE)

class MambaModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = MambaLMHeadModel(self.config).to('cuda')
        
    def forward(self, x):
        x = self.model(x).logits
        return x
    

lr = 3e-4
mamba = MambaModel(config)


class LitMamba(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        batch = batch['input_ids']
        x, y_true = batch[:, :-1], batch[:, 1:]
        
        y_pred = self.model(x) # -> BATCH x SEQ x VOCAB
        
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])
        y_true = y_true.reshape(-1)

        loss = F.cross_entropy(y_pred, y_true)

        self.logger.experiment.add_scalar('Loss', loss, self.trainer.global_step)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(mamba.parameters(), lr=3e-4)
        return optimizer
    

data_loader = DataLoader(dataset=dataset, collate_fn=collator, batch_size=4, num_workers=11)

m = LitMamba(mamba)

checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints', filename='model_{epoch}')

logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = L.Trainer(max_epochs=150, devices=1, logger=logger)

trainer.fit(model=m, train_dataloaders=data_loader)