import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.encoding[:, :seq_len, :].to(x.device)
        return x

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        
        # Cross-attention layers
        self.cross_attn_layer1 = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.cross_attn_layer2 = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)

        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, input_features1, input_features2):
        if self.positional_encoding:
            input_features1 = self.positional_encoding(input_features1)
            input_features2 = self.positional_encoding(input_features2)
        
        x1 = self.sa_decoder(input_features1)
        x2 = self.sa_decoder(input_features2)
        
        # Cross-attention mechanism
        x1, _ = self.cross_attn_layer1(x1, x2, x2)
        x2, _ = self.cross_attn_layer2(x2, x1, x1)
        
        # Flatten and pass through output layers
        x1 = x1.reshape(x1.shape[0], -1)
        x2 = x2.reshape(x2.shape[0], -1)
        
        for layer in self.output_layers:
            x1 = F.relu(layer(x1))
            x2 = F.relu(layer(x2))

        return x1, x2

# 使用示例
class YourModel(nn.Module):
    def __init__(self, context_size=4):
        super(YourModel, self).__init__()
        self.context_size = context_size
        self.shared_decoder = MultiLayerDecoder(embed_dim=256, seq_len=context_size+3, output_layers=[256, 128, 64], nhead=4, num_layers=2, ff_dim_factor=4)

    def forward(self, input_features):
        velocity_predictions, waypoint_predictions = self.shared_decoder(input_features, input_features)
        velocity_predictions = velocity_predictions.view(-1, 4, 2)  # 预测速度序列
        waypoint_predictions = waypoint_predictions.view(-1, 4, 2)  # 预测航路点序列
        return velocity_predictions, waypoint_predictions
