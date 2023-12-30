import numpy as np
import gradio as gr
import os
import torch
import torch.nn as nn
import random
import pickle
import datetime
from torch.nn import functional as F
from modules import processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_iters = 100

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(script_directory, 'models')
data_directory = os.path.join(script_directory, 'data')
output_directory = os.path.join(script_directory, 'outputs')

for directory in [model_directory, data_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        
model_files = [f for f in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, f)) and f.endswith('.pkl')]
data_folders = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]

def refresh_data():
    global data_folders
    data_folders = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
    return gr.Dropdown.update(choices=data_folders)
    
def refresh_models():
    global model_files
    model_files = [f for f in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, f)) and f.endswith('.pkl')]
    return gr.Dropdown.update(choices=model_files)

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
    
@torch.no_grad()    
def estimate_loss(model, train_data, test_data, batch_size):
    results = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == 'train':
                batch_input, batch_target = get_batch(train_data, model.sequence_length, batches=batch_size)
            else:
                batch_input, batch_target = get_batch(test_data, model.sequence_length, batches=batch_size)
            logits, loss = model(batch_input, batch_target)
            losses[k] = loss.item()
        results[split] = losses.mean()
    model.train()
    return results
    
    
def get_batch(data, sequence_length, batches=1):
    indices = torch.randint(len(data) - sequence_length, (batches,))
    batch_input = torch.stack([data[i:i + sequence_length] for i in indices])
    batch_target = torch.stack([data[i + 1:i + sequence_length + 1] for i in indices])
    return batch_input, batch_target


def prepare_data(midi_folder, name, progress=gr.Progress(track_tqdm=True)):
    result_path = os.path.join(data_directory, name)
    midi_files = [file for file in os.listdir(midi_folder) if file.endswith(".mid")]

    encoded = []
    train_data = []
    test_data = []

    for midi_file in progress.tqdm(midi_files):
        midi_file_path = os.path.join(midi_folder, midi_file)
        print(f"Processing file {midi_file_path}")
        encoded += processor.encode_midi(midi_file_path)
        split_index = int((0.8 * len(encoded)))
        train_data += encoded[:split_index]
        test_data += encoded[split_index:]

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    train_path = os.path.join(result_path, 'train.npy')
    test_path = os.path.join(result_path, 'test.npy')
    
    np.save(train_path, train_data)
    np.save(test_path, test_data)
    return "Done"


def train_model(model_choice, data_choice, num_batches, num_iters, learning_rate, progress=gr.Progress(track_tqdm=True)):
    data_path = os.path.join(data_directory, data_choice)
    model_path = os.path.join(model_directory, model_choice)
    num_batches = int(num_batches)
    num_iters = int(num_iters)
    learning_rate = float(learning_rate)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    m = model.to(device)
    train_data = np.load(os.path.join(data_path, 'train.npy'))
    test_data = np.load(os.path.join(data_path, 'test.npy'))
    train_data = torch.tensor(train_data, dtype=torch.long).to(device)
    test_data = torch.tensor(test_data, dtype=torch.long).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in progress.tqdm(range(num_iters)):
        if iter % eval_iters == 0:
            losses = estimate_loss(m, train_data, test_data, num_batches)
            print(f"step: {iter}, train loss: {losses['train']:.3f}, test loss: {losses['test']:.3f}")
        
        batch_input, batch_target = get_batch(train_data, m.sequence_length, num_batches)
        logits, loss = m.forward(batch_input, batch_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    with open(model_path, 'wb') as f:
        pickle.dump(m, f)
    
    final_losses = estimate_loss(m, train_data, test_data, num_batches)
    output = (f"train loss: {final_losses['train']:.3f}, test loss: {final_losses['test']:.3f}")
    
    return output


def create_model(data_choice, model_name, batch_size, sequence_length, learning_rate, n_emb, n_head, n_layer, dropout_value, initial_iters, progress=gr.Progress(track_tqdm=True)):
    data_path = os.path.join(data_directory, data_choice)
    model_output_path = os.path.join(model_directory, model_name)
    sequence_length = int(sequence_length)
    learning_rate = float(learning_rate)
    n_emb = int(n_emb)
    n_head = int(n_head)
    n_layer = int(n_layer)
    dropout = float(dropout_value)
    
    model = MIDIGPT(388, n_emb, n_head, n_layer, sequence_length, dropout)
    m = model.to(device)
    
    batch_size = int(batch_size)
    iters = int(initial_iters)
    train_data = np.load(os.path.join(data_path, 'train.npy'))
    test_data = np.load(os.path.join(data_path, 'test.npy'))
    train_data = torch.tensor(train_data, dtype=torch.long).to(device)
    test_data = torch.tensor(test_data, dtype=torch.long).to(device)  

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in progress.tqdm(range(iters)):
        if iter % eval_iters == 0:
            losses = estimate_loss(m, train_data, test_data, batch_size)
            print(f"step: {iter}, train loss: {losses['train']:.3f}, test loss: {losses['test']:.3f}")
        
        batch_input, batch_target = get_batch(train_data, m.sequence_length, batch_size)
        logits, loss = m.forward(batch_input, batch_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    with open(f"{model_output_path}.pkl", 'wb') as f:
        pickle.dump(m, f)
        
    final_losses = estimate_loss(m, train_data, test_data, batch_size)
    output = (f"train loss: {final_losses['train']:.3f}, test loss: {final_losses['test']:.3f}")
        
    return output


def generate_notes(model_choice, note_amount):
    model_name, model_extension = os.path.splitext(model_choice)
    model_path = os.path.join(model_directory, model_choice)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    m = model.to(device)
    note_amount = int(note_amount)
    random_seed = random.randint(0, 388)
    context_vector = [random_seed]
    context_tensor = torch.tensor(context_vector, dtype=torch.long).to(device)
    generated_output = m.generate(context_tensor.unsqueeze(0), new_tokens_amount=note_amount)[0].tolist()
    current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_name = f"{current_datetime}_{model_name}"
    output_path = os.path.join(output_directory, output_name + '.mid')
    processor.decode_midi(generated_output, file_path=output_path)
    
    return f"Saved as {output_path}"



with gr.Blocks() as gui:
    gr.Markdown("Example of GUI in gradio for MIDI AI")
    with gr.Tab("Prepare Data"):
        data_location = gr.Textbox(label="Dataset location")
        dataset_name = gr.Textbox(label="Dataset name")
        prepare_button = gr.Button("Prepare")
        output_box = gr.Textbox(label="Output")
    
    with gr.Tab("New Model"):
        with gr.Row():
            dataset_selection = gr.Dropdown(choices=data_folders, label="Select dataset")
            refresh_dataset_button = gr.Button("Refresh")
        model_name = gr.Textbox(label="Model Name")
        with gr.Row():
            batch_size = gr.Textbox("32", label="Batch Size")
            sequence_length = gr.Textbox("256", label="Block Size")
            learning_rate = gr.Textbox("1e-3", label="Learning Rate")
            embedding_size = gr.Textbox("400", label="Embedding Size")
        with gr.Row():
            num_heads = gr.Textbox("16", label="Amount of Heads")
            num_layers = gr.Textbox("8", label="Amount of Layers")
            dropout_value = gr.Textbox("0.2", label="Droput")
            initial_iterations = gr.Textbox("1000", label="Amount of Iterations")
        create_button = gr.Button("Create")
        creation_output = gr.Textbox(label="Output")
    
    with gr.Tab("Training"):
        with gr.Row():
            model_choice_training = gr.Dropdown(choices=model_files, label="Select a model")
            refresh_model_button = gr.Button("Refresh")
        with gr.Row():
            data_choice_training = gr.Dropdown(choices=data_folders, label="Select dataset")
            refresh_data_button = gr.Button("Refresh")
        with gr.Row():
            num_batches = gr.Textbox("16", label="Number of batches")
            num_iterations = gr.Textbox("100", label="Number of iterations")
            learning_rate_training = gr.Textbox("1e-3", label="Learning Rate")
        train_button = gr.Button("Train")
        training_output = gr.Textbox(label="Output")
        
    with gr.Tab("Generation"):
        with gr.Row():
            model_choice_generation = gr.Dropdown(choices=model_files, label="Select a model")
            refresh_generation_button = gr.Button("Refresh")
        notes_amount = gr.Textbox("1024", label="Amount of notes")
        generate_button = gr.Button("Generate")
        generation_output = gr.Textbox(label="Output")
        
        
    prepare_button.click(prepare_data, inputs=[data_location, dataset_name], outputs=output_box)
    create_button.click(create_model, inputs=[dataset_selection, model_name, batch_size, sequence_length, learning_rate, embedding_size, num_heads, num_layers, dropout_value, initial_iterations], outputs=creation_output)
    train_button.click(train_model, inputs=[model_choice_training, data_choice_training, num_batches, num_iterations, learning_rate_training], outputs=training_output)
    generate_button.click(generate_notes, inputs=[model_choice_generation, notes_amount], outputs=generation_output)
    
    refresh_dataset_button.click(refresh_data, outputs=dataset_selection)
    refresh_model_button.click(refresh_models, outputs=model_choice_training)
    refresh_data_button.click(refresh_data, outputs=data_choice_training)
    refresh_generation_button.click(refresh_models, outputs=model_choice_generation)

gui.queue().launch()