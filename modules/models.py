import torch
import torch.nn as nn
from torch.nn import functional as F

class MIDIGPT(nn.Module):
    class AttentionHead(nn.Module):
        def __init__(self, embedding_size, head_size, sequence_length, dropout_rate):
            super().__init__()
            self.key_transform = nn.Linear(embedding_size, head_size, bias=False)
            self.query_transform = nn.Linear(embedding_size, head_size, bias=False)
            self.value_transform = nn.Linear(embedding_size, head_size, bias=False)
            lower_triangular_mask = torch.tril(torch.ones(sequence_length, sequence_length))
            self.register_buffer('lower_triangular_mask', lower_triangular_mask)
            self.dropout_layer = nn.Dropout(dropout_rate)

        def forward(self, input_tensor):
            batch_size, seq_len, _ = input_tensor.shape
            key = self.key_transform(input_tensor)
            query = self.query_transform(input_tensor)
            attention_scores = query @ key.transpose(-2, -1) * key.shape[-1]**-0.5
            attention_scores = attention_scores.masked_fill(self.lower_triangular_mask[:seq_len, :seq_len] == 0, float('-inf'))
            normalized_scores = F.softmax(attention_scores, dim=-1)
            normalized_scores = self.dropout_layer(normalized_scores)
            value = self.value_transform(input_tensor)
            output = normalized_scores @ value
            return output

    class MultiHeadAttention(nn.Module):
        def __init__(self, embedding_size, num_heads, head_size, sequence_length, dropout_rate):
            super().__init__()
            self.heads = nn.ModuleList([MIDIGPT.AttentionHead(embedding_size, head_size, sequence_length, dropout_rate) for _ in range(num_heads)])
            self.output_projection = nn.Linear(head_size * num_heads, embedding_size)
            self.dropout_layer = nn.Dropout(dropout_rate)

        def forward(self, input_tensor):
            concatenated_heads = torch.cat([head(input_tensor) for head in self.heads], dim=-1)
            projected_output = self.dropout_layer(self.output_projection(concatenated_heads))
            return projected_output

    class FeedForward(nn.Module):
        def __init__(self, embedding_size, dropout_rate):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(embedding_size, 4 * embedding_size),
                nn.GELU(),
                nn.Linear(4 * embedding_size, embedding_size),
                nn.Dropout(dropout_rate),
            )

        def forward(self, input_tensor):
            return self.network(input_tensor)

    class Block(nn.Module):
        def __init__(self, embedding_size, num_heads, sequence_length, dropout_rate):
            super().__init__()
            head_size = embedding_size // num_heads
            self.self_attention = MIDIGPT.MultiHeadAttention(embedding_size, num_heads, head_size, sequence_length, dropout_rate)
            self.feed_forward = MIDIGPT.FeedForward(embedding_size, dropout_rate)
            self.layer_norm_1 = nn.LayerNorm(embedding_size)
            self.layer_norm_2 = nn.LayerNorm(embedding_size)

        def forward(self, input_tensor):
            attention_output = self.self_attention(input_tensor)
            add_and_norm_1 = self.layer_norm_1(input_tensor + attention_output)
            feed_forward_output = self.feed_forward(add_and_norm_1)
            add_and_norm_2 = self.layer_norm_2(add_and_norm_1 + feed_forward_output)
            return add_and_norm_2

    def __init__(self, vocab_size, embedding_size, num_heads, num_layers, sequence_length, dropout_rate):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size).to(self.device)
        self.position_embedding_table = nn.Embedding(sequence_length, embedding_size).to(self.device)
        self.blocks = nn.Sequential(*[MIDIGPT.Block(embedding_size, num_heads, sequence_length, dropout_rate) for _ in range(num_layers)]).to(self.device)
        self.layer_norm_final = nn.LayerNorm(embedding_size).to(self.device)
        self.language_model_head = nn.Linear(embedding_size, vocab_size).to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, indices, targets=None):
        indices = indices.to(self.device)
        batch_size, sequence_length = indices.shape
        token_embeddings = self.token_embedding_table(indices)
        position_embeddings = self.position_embedding_table(torch.arange(sequence_length, device=self.device))
        combined_embeddings = token_embeddings + position_embeddings
        transformed_embeddings = self.blocks(combined_embeddings)
        normalized_embeddings = self.layer_norm_final(transformed_embeddings)
        logits = self.language_model_head(normalized_embeddings)

        if targets is not None:
            targets = targets.to(self.device)
            batch_size, sequence_length, embedding_size = logits.shape
            logits_flattened = logits.view(batch_size * sequence_length, embedding_size)
            targets_flattened = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits_flattened, targets_flattened)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, index, new_tokens_amount):
        index = index.to(self.device)
        self.eval()
        for _ in range(new_tokens_amount):
            if index.size()[1] < self.sequence_length:
                logits, loss = self.forward(index)
            else:
                logits, loss = self.forward(index[:, -(self.sequence_length - 1):])
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probabilities, num_samples=1)
            index = torch.cat((index, next_index), dim=1)
        self.train()
        return index
    
class MIDILSTM(nn.Module):
    def __init__(self, vocab_size, n_emb, hidden_size, num_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_emb = n_emb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.token_embedding = nn.Embedding(vocab_size, n_emb)
        self.lstm = nn.LSTM(n_emb, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.to(self.device)

    def forward(self, idx, targets=None):
        embeddings = self.token_embedding(idx).to(self.device)
        lstm_out, _ = self.lstm(embeddings)
        logits = self.fc(lstm_out)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, start_idx, new_tokens_amount):
        self.eval()
        generated = start_idx.to(self.device)

        for _ in range(new_tokens_amount):
            embeddings = self.token_embedding(generated)
            lstm_out, _ = self.lstm(embeddings)
            logits = self.fc(lstm_out[:, -1, :])
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            
        self.train()
        return generated