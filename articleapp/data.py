from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from torch.utils.data import DataLoader
import torch
import gluonnlp as nlp
from articleapp.dataset import BERTDataset

class Data():
  def bert(self):
    bert_model, _ = get_pytorch_kobert_model()
    return bert_model

  def tokenizer(self):
    _ , vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    return tok

  def make_dataset(self, dataset_train, tok, max_len):
    data = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    return data

  def make_dataloader(self, data, batch_size):
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=0)
    return dataloader