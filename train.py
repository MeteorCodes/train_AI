# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import logging
from collections import Counter
import random
import sys

# 配置日志
logging.basicConfig(
    filename='dialogue_training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 强制使用CPU
device = torch.device('cpu')
print(f"使用设备: {device}")
logging.info(f"使用设备: {device}")

# 模型参数
MAX_VOCAB_SIZE = 1000
EMBEDDING_DIM = 64
HIDDEN_DIM = 512
NUM_LAYERS = 10
BATCH_SIZE = 16
EPOCHS = 200
VAL_SPLIT = 0.1


class DialogueDataset(Dataset):
    """
    处理对话数据，包括加载、分词和构建词汇表。
    """

    def __init__(self, data_path, vocab=None, max_vocab_size=1000, val_split=0.1):
        self.data_path = data_path
        self.max_vocab_size = max_vocab_size
        self.val_split = val_split
        self.data = self._load_data()
        self.input_texts, self.target_texts = self._prepare_data()
        self.vocab = vocab if vocab else self._build_vocab()
        self.word2idx = self.vocab['word2idx']
        self.idx2word = self.vocab['idx2word']
        self.train_data, self.val_data = self._split_data()

    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data = [line.strip().split('|') for line in lines if '|' in line]
        return data

    def _prepare_data(self):
        input_texts = [item[0] for item in self.data]
        target_texts = [item[1] for item in self.data]
        return input_texts, target_texts

    def _tokenize(self, text):
        # 针对中文的简单字符级别分词
        return [char for char in text]

    def _build_vocab(self):
        all_tokens = []
        for text in self.input_texts + self.target_texts:
            all_tokens.extend(self._tokenize(text))

        counter = Counter(all_tokens)
        sorted_vocab = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        word2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        idx2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}

        for word, _ in sorted_vocab[:self.max_vocab_size - 4]:
            if word not in word2idx:
                idx = len(word2idx)
                word2idx[word] = idx
                idx2word[idx] = word

        return {'word2idx': word2idx, 'idx2word': idx2word}

    def _split_data(self):
        data_pairs = list(zip(self.input_texts, self.target_texts))
        random.shuffle(data_pairs)
        val_size = int(len(data_pairs) * self.val_split)
        val_data = data_pairs[:val_size]
        train_data = data_pairs[val_size:]
        return train_data, val_data

    def __len__(self):
        return len(self.data)

    def get_tokenized_data(self, is_train=True):
        data = self.train_data if is_train else self.val_data
        tokenized_data = []
        for input_text, target_text in data:
            input_tokens = [self.word2idx.get(t, self.word2idx['<UNK>']) for t in self._tokenize(input_text)]
            target_tokens = [self.word2idx['<SOS>']] + [self.word2idx.get(t, self.word2idx['<UNK>']) for t in
                                                        self._tokenize(target_text)] + [self.word2idx['<EOS>']]
            tokenized_data.append(
                (torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)))
        return tokenized_data

    @staticmethod
    def pad_collate(batch):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        return inputs, targets


# 模型定义
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=False,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden, cell, encoder_outputs=None):
        embedded = self.embedding(x.unsqueeze(1))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.MAX_LENGTH = 100

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        max_len = target.shape[1]
        target_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(max_len, batch_size, target_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(source)
        decoder_input = target[:, 0]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = target[:, t] if teacher_force and t < target.shape[1] else top1
        return outputs


def train_model(model, train_loader, optimizer, criterion, clip_norm=1.0):
    model.train()
    total_loss = 0
    with tqdm(train_loader, desc="训练中") as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, targets)
            outputs = outputs[1:].view(-1, outputs.shape[-1])
            targets = targets[:, 1:].contiguous().view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(train_loader)


def validate_model(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(val_loader, desc="验证中") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, targets, 0)  # 无教师强制
                outputs = outputs[1:].view(-1, outputs.shape[-1])
                targets = targets[:, 1:].contiguous().view(-1)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(val_loader)


if __name__ == "__main__":
    try:
        # 数据加载和词汇表构建
        dataset = DialogueDataset('dialogue_data.txt', max_vocab_size=MAX_VOCAB_SIZE, val_split=VAL_SPLIT)
        train_data = dataset.get_tokenized_data(is_train=True)
        val_data = dataset.get_tokenized_data(is_train=False)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dataset.pad_collate)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dataset.pad_collate)

        # 初始化模型、优化器和损失函数
        encoder = Encoder(len(dataset.word2idx), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
        decoder = Decoder(len(dataset.word2idx), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
        model = Seq2Seq(encoder, decoder, device).to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD标记

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        best_val_loss = float('inf')
        model_dir = 'trained_models'
        os.makedirs(model_dir, exist_ok=True)

        # 训练循环
        for epoch in range(EPOCHS):
            avg_train_loss = train_model(model, train_loader, optimizer, criterion)
            avg_val_loss = validate_model(model, val_loader, criterion)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch + 1}/{EPOCHS} - 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f}")
            logging.info(f"Epoch {epoch + 1} - 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'vocab': {
                        'word2idx': dataset.word2idx,
                        'idx2word': dataset.idx2word
                    }
                }, os.path.join(model_dir, 'best_model.pth'))
                print("保存最佳模型!")
                logging.info("保存最佳模型!")

    except FileNotFoundError:
        print("错误: 未找到训练数据文件 'dialogue_data.txt'。")
        print("请创建一个包含对话数据的 'dialogue_data.txt' 文件。")
        sys.exit(1)
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        sys.exit(1)
