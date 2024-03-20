import numpy as np
import gradio as gr
import os
import torch
import random
import pickle
import time
import datetime
from modules import processor
from modules.models import MIDIGPT, MIDILSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_ITERS = 100

script_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = os.path.join(script_directory, 'models')
data_directory = os.path.join(script_directory, 'data')
output_directory = os.path.join(script_directory, 'outputs')

for directory in [model_directory, data_directory]:
    if not os.path.exists(directory):
        os.makedirs(directory)

model_files = [
    f for f in os.listdir(model_directory)
    if os.path.isfile(os.path.join(model_directory, f)) and f.endswith('.pkl')
]
data_folders = [
    f for f in os.listdir(data_directory)
    if os.path.isdir(os.path.join(data_directory, f))
]

def refresh_data():
    global data_folders
    data_folders = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
    return gr.Dropdown.update(choices=data_folders)

def refresh_models():
    global model_files
    model_files = [f for f in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, f)) and f.endswith('.pkl')]
    return gr.Dropdown.update(choices=model_files)


@torch.no_grad()    
def estimate_loss(model, train_data, test_data, batch_size):
    results = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            if split == 'train':
                if isinstance(model, MIDILSTM):
                    batch_input, batch_target = get_batch(train_data, 512, batches=batch_size)
                else:
                    batch_input, batch_target = get_batch(train_data, model.sequence_length, batches=batch_size)
            else:
                if isinstance(model, MIDILSTM):
                    batch_input, batch_target = get_batch(test_data, 512, batches=batch_size)
                else:
                    batch_input, batch_target = get_batch(test_data, model.sequence_length, batches=batch_size)
            _, loss = model(batch_input, batch_target)
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
    midi_files = [file for file in os.listdir(midi_folder) if (file.endswith(".mid") or file.endswith(".midi"))]

    encoded = []
    train_data = []
    test_data = []
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    else:
        return "Data already exists. Please choose diffrent name or delete previous data."
        
    train_path = os.path.join(result_path, 'train.npy')
    test_path = os.path.join(result_path, 'test.npy')
    
    for iter, midi_file in enumerate(progress.tqdm(midi_files)):
        midi_file_path = os.path.join(midi_folder, midi_file)
        print(f"Processing file {midi_file_path}")
        try:
            encoded = processor.encode_midi(midi_file_path)
        except Exception as e:
            print(f"Exception of type {type(e)} caught: {e}")
        split_index = int((0.8 * len(encoded)))
        train_data += encoded[:split_index]
        test_data += encoded[split_index:]
        if iter % 100 == 0:
            if os.path.exists(train_path):
                train_temp = np.load(train_path)
                test_temp = np.load(test_path)
                train_out = np.concatenate((train_temp, train_data))
                test_out = np.concatenate((test_temp, test_data))
                np.save(train_path, train_out)
                np.save(test_path, test_out)
            else:
                np.save(train_path, train_data)
                np.save(test_path, test_data)
            train_data = []
            test_data = []
            train_temp = []
            test_temp = []
            
    train_temp = np.load(train_path)
    test_temp = np.load(test_path)
    train_out = np.concatenate((train_temp, train_data))
    test_out = np.concatenate((test_temp, test_data))
    np.save(train_path, train_out)
    np.save(test_path, test_out)
    
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

    losses_records = []

    for iter in progress.tqdm(range(num_iters)):
        if iter % EVAL_ITERS == 0:
            losses = estimate_loss(m, train_data, test_data, num_batches)
            print(f"step: {iter}, train loss: {losses['train']:.3f}, test loss: {losses['test']:.3f}")
            losses_records.append({
                'iteration': iter,
                'train_loss': losses['train'].item(),
                'test_loss': losses['test'].item(),
                'num_batches': num_batches,
                'learning_rate': learning_rate
            })
        if isinstance(model, MIDILSTM):
            batch_input, batch_target = get_batch(train_data, 512, num_batches)
        else:
            batch_input, batch_target = get_batch(train_data, m.sequence_length, num_batches)
        _, loss = m.forward(batch_input, batch_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    final_losses = estimate_loss(m, train_data, test_data, num_batches)
    output = (f"train loss: {final_losses['train']:.3f}, test loss: {final_losses['test']:.3f}")
    losses_records.append({
                'iteration': iter + 1,
                'train_loss': losses['train'].item(),
                'test_loss': losses['test'].item(),
                'num_batches': num_batches,
                'learning_rate': learning_rate
            })
    
    with open(model_path, 'wb') as f:
        pickle.dump(m, f)
    
    return output


def create_GPT_model(GPT_model_name, sequence_length, n_emb, n_head, n_layer, dropout_value):
    model_output_path = os.path.join(model_directory, GPT_model_name)
    sequence_length = int(sequence_length)
    n_emb = int(n_emb)
    n_head = int(n_head)
    n_layer = int(n_layer)
    dropout = float(dropout_value)
    
    model = MIDIGPT(388, n_emb, n_head, n_layer, sequence_length, dropout)
        
    with open(f"{model_output_path}.pkl", 'wb') as f:
        pickle.dump(model, f)
        
    return "GPT Model Created"

def create_LSTM_model(LSTM_model_name, hidden_size, embedding_size, num_layers, dropout_value):
    model_output_path = os.path.join(model_directory, LSTM_model_name)
    hidden_size = int(hidden_size)
    embedding_size = int(embedding_size)
    num_layers = int(num_layers)
    dropout = float(dropout_value)
    
    model = MIDILSTM(388, embedding_size, hidden_size, num_layers, dropout)
        
    with open(f"{model_output_path}.pkl", 'wb') as f:
        pickle.dump(model, f)
        
    return "LSTM Model Created"


def generate_tokens(model_choice, token_amount, tracks_amount, progress=gr.Progress(track_tqdm=True)):
    model_name, _ = os.path.splitext(model_choice)
    model_path = os.path.join(model_directory, model_choice)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    m = model.to(device)
    token_amount = int(token_amount)
    tracks_amount = int(tracks_amount)
    outputs = []
    
    for _ in progress.tqdm(range(tracks_amount)):
        random_seed = random.randint(0, 388)
        context_vector = [random_seed]
        context_tensor = torch.tensor(context_vector, dtype=torch.long).to(device)
        generated_output = m.generate(context_tensor.unsqueeze(0), new_tokens_amount=token_amount)[0].tolist()
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        output_name = f"{current_datetime}_{model_name}"
        output_path = os.path.join(output_directory, output_name + '.mid')
        processor.decode_midi(generated_output, file_path=output_path)
        
        message = f"Saved as {output_path}"
        outputs.append(message)
    
    output = "\n".join(outputs)
    return output

def fetch_batch(data_choice_control, amount_to_get, tokens_to_get):
    data_path = os.path.join(data_directory, data_choice_control)
    train_data = np.load(os.path.join(data_path, 'train.npy'))
    train_data = torch.tensor(train_data, dtype=torch.long).to(device)
    amount_to_get = int(amount_to_get)
    tokens_to_get = int(tokens_to_get)
    outputs = []
    
    batches, _ = get_batch(train_data, tokens_to_get, amount_to_get)
    
    for batch in batches:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        time.sleep(1)
        output_name = f"{current_datetime}_{data_choice_control}"
        output_path = os.path.join(output_directory, output_name + '.mid')
        processor.decode_midi(batch.tolist(), file_path=output_path)
        
        message = f"Saved as {output_path}"
        outputs.append(message)
    
    output = "\n".join(outputs)
    return output


with gr.Blocks() as gui:
    gr.Markdown("Example of GUI in gradio for MIDI AI")
    with gr.Tab("Prepare Data"):
        data_location = gr.Textbox(label="Dataset location")
        dataset_name = gr.Textbox(label="Dataset name")
        prepare_button = gr.Button("Prepare")
        output_box = gr.Textbox(label="Output")
    
    with gr.Tab("New GPT Model"):
        with gr.Row():
            GPT_model_name = gr.Textbox(label="GPT Model Name")
        with gr.Row():
            sequence_length = gr.Textbox("512", label="Block Size")
            GPT_embedding_size = gr.Textbox("256", label="Embedding Size")
            num_heads = gr.Textbox("8", label="Amount of Heads")
            GPT_num_layers = gr.Textbox("4", label="Amount of Layers")
            GPT_dropout_value = gr.Textbox("0.2", label="Dropout")
        GPT_create_button = gr.Button("Create")
        GPT_creation_output = gr.Textbox(label="Output")
        
    with gr.Tab("New LSTM Model"):
        with gr.Row():
            LSTM_model_name = gr.Textbox(label="LSTM Model Name")
        with gr.Row():
            hidden_size = gr.Textbox("512", label="Hidden Size")
            LSTM_num_layers = gr.Textbox("4", label="Amount of Layers")
            LSTM_embedding_size = gr.Textbox("256", label="Embedding Size")
            LSTM_dropout_value = gr.Textbox("0.2", label="Dropout")
        LSTM_create_button = gr.Button("Create")
        LSTM_creation_output = gr.Textbox(label="Output")
    
    with gr.Tab("Training"):
        with gr.Row():
            model_choice_training = gr.Dropdown(choices=model_files, label="Select a model")
            refresh_model_button = gr.Button("Refresh")
        with gr.Row():
            data_choice_training = gr.Dropdown(choices=data_folders, label="Select dataset")
            refresh_data_button = gr.Button("Refresh")
        with gr.Row():
            num_batches = gr.Textbox("64", label="Number of batches")
            num_iterations = gr.Textbox("5000", label="Number of iterations")
            learning_rate_training = gr.Textbox("1e-3", label="Learning Rate")
        train_button = gr.Button("Train")
        training_output = gr.Textbox(label="Output")
        
    with gr.Tab("Generation"):
        with gr.Row():
            model_choice_generation = gr.Dropdown(choices=model_files, label="Select a model")
            refresh_generation_button = gr.Button("Refresh")
        with gr.Row():    
            tokens_amount = gr.Textbox("1024", label="Amount of tokens")
            tracks_amount = gr.Textbox("1", label="Amount of tracks")
        generate_button = gr.Button("Generate")
        generation_output = gr.Textbox(label="Output")
        
    with gr.Tab("Control Data"):
        with gr.Row():
            data_choice_control = gr.Dropdown(choices=data_folders, label="Select dataset")
            refresh_dataset_button = gr.Button("Refresh")
        amount_to_get = gr.Textbox("1", label="Amount of tracks")
        tokens_to_get = gr.Textbox("1024", label="Amount of tokens")
        batch_button = gr.Button("Get")
        batch_output = gr.Textbox(label="Output")
    
        
    prepare_button.click(prepare_data, inputs=[data_location, dataset_name], outputs=output_box)
    GPT_create_button.click(create_GPT_model, inputs=[GPT_model_name, sequence_length, GPT_embedding_size, num_heads, GPT_num_layers, GPT_dropout_value], outputs=GPT_creation_output)
    LSTM_create_button.click(create_LSTM_model, inputs=[LSTM_model_name, hidden_size, LSTM_embedding_size, LSTM_num_layers, LSTM_dropout_value], outputs=LSTM_creation_output)
    train_button.click(train_model, inputs=[model_choice_training, data_choice_training, num_batches, num_iterations, learning_rate_training], outputs=training_output)
    generate_button.click(generate_tokens, inputs=[model_choice_generation, tokens_amount, tracks_amount], outputs=generation_output)
    batch_button.click(fetch_batch,inputs=[data_choice_control, amount_to_get, tokens_to_get], outputs=batch_output)
    
    refresh_dataset_button.click(refresh_data, outputs=data_choice_control)
    refresh_model_button.click(refresh_models, outputs=model_choice_training)
    refresh_data_button.click(refresh_data, outputs=data_choice_training)
    refresh_generation_button.click(refresh_models, outputs=model_choice_generation)

gui.queue().launch()