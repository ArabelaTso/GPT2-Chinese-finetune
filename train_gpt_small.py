import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config, get_cosine_schedule_with_warmup
import numpy as np
import math

GPU_ID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('***************GPU_ID***************: ', GPU_ID)
else:
    raise NotImplementedError


class Vocabulary:
    def __init__(self, file_ls):
        self.file_ls = file_ls
        self.word2id = {}
        self.id2word = {}
        self.vocab = {'<PAD>', '<UNK>', '<BOS>', '<EOS>'}
        self.vocab_size = 0
        self.build()

    def build(self):
        for file in self.file_ls:
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read()
            self.vocab = self.vocab.union(set(text.split()))
            for i, word in enumerate(self.vocab):
                self.word2id[word] = i
                self.id2word[i] = word
            self.vocab_size = len(self.vocab)


class TextDataset(Dataset):
    def __init__(self, filepath, seq_length, vocab):
        self.seq_length = seq_length
        self.vocab = vocab
        with open(filepath, 'r', encoding='utf-8') as f:
            self.text = f.read()
        sentences = self.text.split('\n')
        sentences = " <EOS> <BOS> ".join(sentences)
        sentences = "<BOS> " + sentences + " <EOS>"
        with open('toy_data/sentences.txt', 'w', encoding='utf-8') as f:
            f.write(sentences)
        self.text_idx = [self.vocab.word2id[w] for w in sentences.split()]
        self.length = self.__len__()

    def __len__(self):
        return len(self.text_idx) - self.seq_length

    def __getitem__(self, idx):
        inputs = torch.tensor(self.text_idx[idx:idx + self.seq_length])
        targets = torch.tensor(self.text_idx[idx + 1:idx + self.seq_length + 1])
        return inputs, targets


class LanguageModel_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers):
        super(LanguageModel_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None):
        x = self.embedding(x)
        if h0 is not None:
            output, h = self.rnn(x, h0)
        else:
            output, h = self.rnn(x)
        output = self.fc(output.reshape(-1, output.shape[2]))
        return output, h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LanguageModel_transformer_encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead, num_layers, dim_feedforward):
        super(LanguageModel_transformer_encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_encoding = PositionalEncoding(embedding_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_size, nhead, dim_feedforward), num_layers)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = x.permute(1, 0, 2)
        output = self.transformer_encoder(x)
        output = self.fc(output.reshape(-1, output.shape[2]))
        return output


class GPT2(nn.Module):
    def __init__(self, vocab):
        super(GPT2, self).__init__()
        self.vocab = vocab
        configuration = GPT2Config()
        configuration.vocab_size = vocab.vocab_size
        configuration.bos_token_id = vocab.word2id['<BOS>']
        configuration.eos_token_id = vocab.word2id['<EOS>']
        configuration.pad_token_id = vocab.word2id['<PAD>']

        self.transformer = GPT2LMHeadModel(configuration)

    def forward(self, x, y):
        outputs = self.transformer(input_ids=x, labels=y)
        return outputs

    def generate(self, input_ids, max_length):
        output = self.transformer.generate(input_ids=input_ids,
                                           max_length=max_length,
                                           num_beams=5,
                                           no_repeat_ngram_size=2,
                                           early_stopping=False)
        return output


def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                       loss.item()))
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss / len(train_loader)))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('model_try/' + '/final')

def train_gpt(config, model, train_loader, optimizer, scheduler, device):
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(config['num_epochs']):
        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, targets)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['clip'])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, config['num_epochs'],
                                                                         i + 1, len(train_loader), loss.item()))
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, config['num_epochs'], total_loss / len(train_loader)))
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('model_try/' + '/final')

def test(model, test_loader, criterion, vocab, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.reshape(-1))
            total_loss += loss.item()
    print('Test Loss: {:.4f}'.format(total_loss / len(test_loader)))


def generate(model, vocab, start_text, length, seq_length, temperature, device):
    model.eval()
    with torch.no_grad():
        input_text = start_text
        for _ in range(length):
            inputs = torch.tensor([vocab.word2id[w] for w in input_text[-seq_length:].split()],
                                  dtype=torch.long).unsqueeze(0).to(device)
            outputs = model(inputs)
            predictions = F.softmax(outputs[-1] / temperature, dim=0).cpu().numpy()
            sampled_idx = np.random.choice(len(vocab.vocab), p=predictions)
            sampled_char = vocab.id2word[sampled_idx]
            input_text = input_text + ' ' + sampled_char
        print(input_text)
        return input_text


def gpt_portal():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'seq_length': 100,
        'batch_size': 128,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'weight_decay': 0.01,
        'clip': 1.0,
    }

    vocab = Vocabulary(['toy_data/train.txt', 'toy_data/test.txt'])
    print('Vocabulary size: {}'.format(vocab.vocab_size))
    print(vocab.word2id['<BOS>'])
    train_dataset = TextDataset('toy_data/train.txt', config['seq_length'], vocab)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    model = GPT2(vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config['learning_rate'],
                                  eps=1e-8,
                                  weight_decay=config['weight_decay'])
    total_steps = (train_dataset.length // config['batch_size']) * config['num_epochs']
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    train_gpt(config, model, train_loader, optimizer, scheduler, device)
    while True:
        try:
            print('------------------')
            start_text = input('Input: ')
            outputs = model.generate(
                input_ids=torch.tensor([[vocab.word2id[w] for w in start_text.split()]]).to(device), max_length=20)
            cur_ids = outputs[0].tolist()
            cur_text = " ".join([vocab.id2word[x] for x in cur_ids])
            print(cur_text)
        except Exception as e:
            print(e)


def main_portal():
    seq_length = 100
    batch_size = 32
    embedding_size = 256
    nhead = 8
    num_layers = 4
    dim_feedforward = 512
    learning_rate = 0.001
    num_epochs = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab = Vocabulary(['cache/vocab.txt', 'toy_data/test.txt'])
    train_dataset = TextDataset('data/train_all.txt', seq_length, vocab)
    test_dataset = TextDataset('data/train_all.txt', seq_length, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LanguageModel_transformer_encoder(vocab.vocab_size, embedding_size, nhead, num_layers,
                                              dim_feedforward).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train(model, train_loader, criterion, optimizer, num_epochs, device)
    test(model, test_loader, criterion, vocab, device)
    while True:
        start_text = input('Input: ')
        try:
            generate(model, vocab, '<BOS> ' + start_text, 15, seq_length, 1, device)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # gpt_portal()
    main_portal()