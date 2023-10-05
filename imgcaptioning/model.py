import torch
import torch.nn as nn
import torchvision.models as models

# Encoder
class CNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        for param in resnet50.parameters():
            param.requires_grad_(False)

        layers = list(resnet50.children())[:-1]
        self.resnet = nn.Sequential(*layers)
        self.embed = nn.Linear(resnet50.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features

# ----------- Decoder ------------
class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, cap_tokens):
        cap_embedding = self.embed(cap_tokens[:, :-1])
        embeddings = torch.concat((features.unsqueeze(dim=1), cap_embedding), dim=1)
        lstm_out, self.states = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs
    
    def sample(self, inputs, states=None, max_len=20):
        res = []

        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(dim=1))
            _, word_idx = outputs.max(dim=1)
            res.append(word_idx.item())
            if word_idx == 1:
                break
            inputs = self.embed(word_idx)
            inputs = inputs.unsqueeze(1)
        return res