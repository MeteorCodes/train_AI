# inference.py

import torch
import torch.nn as nn
import os
import sys
from collections import Counter  # 需要这个才能重建词汇表
import random

# 模型参数 (必须与训练时的参数一致)
EMBEDDING_DIM = 64
HIDDEN_DIM = 512
NUM_LAYERS = 10
device = torch.device('cpu')
MAX_LENGTH = 100


# 模型定义（与训练时一致）
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


class DialoguePredictor:
    def __init__(self, model_path='./trained_models/best_model.pth'):
        self.device = device
        self.model_path = model_path
        self.word2idx = None
        self.idx2word = None
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.word2idx = checkpoint['vocab']['word2idx']
        self.idx2word = checkpoint['vocab']['idx2word']

        encoder = Encoder(len(self.word2idx), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
        decoder = Decoder(len(self.word2idx), EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)

        model = Seq2Seq(encoder, decoder, self.device)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.eval()

        return model

    def _tokenize(self, text):
        # 针对中文的简单字符级别分词
        return [char for char in text]

    def predict(self, input_text):
        tokens = self._tokenize(input_text)
        input_tensor = torch.tensor([self.word2idx.get(t, self.word2idx['<UNK>']) for t in tokens], dtype=torch.long,
                                    device=self.device).unsqueeze(0)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(input_tensor)

            decoder_input = torch.tensor([self.word2idx['<SOS>']], dtype=torch.long, device=self.device)
            output_tokens = []

            for _ in range(self.model.MAX_LENGTH):
                output, hidden, cell = self.model.decoder(decoder_input, hidden, cell, encoder_outputs)
                predicted_idx = output.argmax().item()

                if predicted_idx == self.word2idx['<EOS>']:
                    break

                output_tokens.append(self.idx2word.get(predicted_idx, '<UNK>'))
                decoder_input = torch.tensor([predicted_idx], device=self.device)

        return ''.join(output_tokens)


if __name__ == "__main__":
    try:
        predictor = DialoguePredictor()
        print("===== 文字对话AI模型 =====")
        print("提示: 输入你的问题，输入 'exit' 退出程序")

        while True:
            user_input = input("你: ")
            if user_input.lower() == 'exit':
                break

            result = predictor.predict(user_input)
            print(f"AI: {result}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 train.py 训练模型并生成 best_model.pth 文件。")
    except Exception as e:
        print(f"发生错误: {e}")
