import logging
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
import datetime
import os
from icecream import ic
import math 
from torch import optim
from datamaestro import prepare_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()

conf = OmegaConf.load('config/config_imdb.yaml')

PAD_ID = 40001

class FolderText(Dataset):
    """Dataset basé sur des dossiers (un par classe) et fichiers"""

    def __init__(self, classes, folder: Path, tokenizer, load=False):
        self.tokenizer = tokenizer
        self.files = []
        self.filelabels = []
        self.labels = {}
        for ix, key in enumerate(classes):
            self.labels[key] = ix

        for label in classes:
            for file in (folder / label).glob("*.txt"):
                self.files.append(file.read_text() if load else file)
                self.filelabels.append(self.labels[label])

    def __len__(self):
        return len(self.filelabels)

    def __getitem__(self, ix):
        s = self.files[ix]
        return self.tokenizer(s if isinstance(s, str) else s.read_text()), self.filelabels[ix]
    
def collate(batch):
    data = [torch.LongTensor(b[0]) for b in batch]
    lens = [len(b[0]) for b in batch]
    labels = [b[1] for b in batch]
    return torch.nn.utils.rnn.pad_sequence(data, padding_value = PAD_ID, batch_first=True), torch.LongTensor(labels)

def get_imdb_data(embedding_size=50):
    """Renvoie l'ensemble des donnéees nécessaires pour l'apprentissage

    - dictionnaire word vers ID
    - embeddings (Glove)
    - DataSet (FolderText)

    """
    WORDS = re.compile(r"\S+")

    words, embeddings = prepare_dataset('edu.stanford.glove.6b.%d' % embedding_size).load()
    OOVID = len(words)
    words.append("__OOV__")

    word2id = {word: ix for ix, word in enumerate(words)}
    embeddings = np.vstack((embeddings, np.zeros(embedding_size)))

    def tokenizer(t):
        return [word2id.get(x, OOVID) for x in re.findall(WORDS, t.lower())]

    logging.info("Loading embeddings")

    logging.info("Get the IMDB dataset")
    ds = prepare_dataset("edu.stanford.aclimdb")

    return word2id, embeddings, FolderText(ds.train.classes, ds.train.path, tokenizer, load=False), FolderText(ds.test.classes, ds.test.path, tokenizer, load=False)




