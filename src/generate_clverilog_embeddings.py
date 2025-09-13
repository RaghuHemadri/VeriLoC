import os
import torch
import argparse
import sys
import pickle

import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import *

def load_cl_verilog_model():
    model_name = 'ajn313/cl-verilog-1.0'
    hugging_face_token = os.getenv('HUGGING_FACE_TOKEN')

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        token = hugging_face_token
    )
    tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(model_name, token = hugging_face_token).to('cuda')

    return model, tokenizer

def module_attention_mask_based_pooling(hidden_states, attention_mask):
    # Reshape attention mask to [batch_size, seq_len, 1] for broadcasting
    attention_mask = attention_mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]

    # Apply the attention mask to the hidden states (mask out padding tokens)
    masked_hidden_states = hidden_states * attention_mask  # Shape: [batch_size, seq_len, hidden_dim]

    # Sum over the sequence dimension (only valid tokens contribute to the sum)
    sum_hidden_states = masked_hidden_states.sum(dim=1)  # Shape: [batch_size, hidden_dim]

    # Compute the sum of the attention mask (i.e., number of valid tokens in each sequence)
    valid_token_count = attention_mask.sum(dim=1)  # Shape: [batch_size, 1]
    
    return sum_hidden_states, valid_token_count

def line_attention_mask_based_pooling(hidden_states, attention_mask):
    # Compute the weighted sum of the tensor
    sum_hidden_states = torch.sum(hidden_states * attention_mask.unsqueeze(1), dim=0)
    
    # Compute the sum of the mask
    valid_token_count = torch.sum(attention_mask)
    
    return sum_hidden_states, valid_token_count

def batch_with_indices(data, batch_size=8):
    """
    Splits a list of strings into batches along with their indices.

    Args:
        data (list): List of strings to batch.
        batch_size (int): The size of each batch.

    Returns:
        tuple: A tuple containing two lists:
               - batches (list): List of batches, where each batch is a list of strings.
               - indices (list): List of batches of indices corresponding to the strings.
    """
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    indices = [list(range(i, min(i + batch_size, len(data)))) for i in range(0, len(data), batch_size)]
    return batches, indices

def get_context_lines(verilog_code_lines, window_size = 0):
    context_verilog_lines = []
    len_ = len(verilog_code_lines)

    for i in range(len_):
        start = max(0, i - window_size)
        end = min(len_, i + window_size + 1)
        # end = i+1

        context = "".join(verilog_code_lines[start:end])
        context_verilog_lines.append(context)

    return context_verilog_lines

def save_module_embeddings(processed_files, pooled_hidden_states_list, embeddings_save_path):
    # Save the hidden state vectors with processed file as key
    hidden_states_dict = {text: pooled_hidden_states.numpy() for text, pooled_hidden_states in zip(processed_files, pooled_hidden_states_list)}

    # Save the dictionary to a file (e.g., using numpy's savez)
    np.savez(embeddings_save_path, **hidden_states_dict)


def get_module_embeddings(model, tokenizer, pooling_fn, relative_verilog_paths, preprocessors):

    embeddings_save_path = '../embeddings/hidden_states_modules.npz'
    pooled_hidden_states_list = []
    processed_files = []
    token_count = []

    num_training_steps = len(relative_verilog_paths)
    progress_bar = tqdm(range(num_training_steps))

    count = 0

    for verilog_file_path in relative_verilog_paths:
        try:
            verilog_code = read_file(verilog_file_path)
            preprocessed_verilog_code = verilog_code
            # preprocessed_verilog_code = preprocess_code(verilog_code, preprocessors)

            inputs = tokenizer(preprocessed_verilog_code, return_tensors="pt", padding=True).to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')
            input_ids = inputs['input_ids'].to('cuda')
            
            split_factor = 1
            success = False
            
            while not success:
                try:
                    if split_factor > len(input_ids[0]) or split_factor > 64:
                        raise RuntimeError("Unable to process input due to excessive memory requirements.")

                    # Calculate split size based on current split factor
                    split_size = len(input_ids[0]) // split_factor
                    split_inputs = [input_ids[0][i:i+split_size] for i in range(0, len(input_ids[0]), split_size)]
                    split_attention_masks = [attention_mask[0][i:i+split_size] for i in range(0, len(attention_mask[0]), split_size)]

                    # Initialize cumulative sums for pooling
                    cumulative_sum_hidden_states = None
                    cumulative_valid_token_count = 0
                    
                    with torch.no_grad():
                        for split_input, split_mask in zip(split_inputs, split_attention_masks):
                            split_input = split_input.unsqueeze(0).to('cuda')
                            split_mask = split_mask.unsqueeze(0).to('cuda')
                            
                            outputs = model(input_ids=split_input, attention_mask=split_mask, output_hidden_states=True)
                            hidden_states = outputs.hidden_states[-1]
                            
                            # Get sum_hidden_states and valid_token_count for the split
                            sum_hidden_states, valid_token_count = pooling_fn(hidden_states, split_mask)
                            
                            if cumulative_sum_hidden_states == None:
                                cumulative_sum_hidden_states = sum_hidden_states.to('cpu')  
                                cumulative_valid_token_count = valid_token_count.to('cpu')
                            else:
                                # Accumulate sum_hidden_states and valid_token_count across splits
                                cumulative_sum_hidden_states += sum_hidden_states.to('cpu')
                                cumulative_valid_token_count += valid_token_count.to('cpu')

                    # Compute overall pooled hidden states by dividing cumulative sum by cumulative valid token count
                    cumulative_valid_token_count = cumulative_valid_token_count.clamp(min=1e-9)
                    pooled_hidden_states = cumulative_sum_hidden_states / cumulative_valid_token_count
                    pooled_hidden_states_list.append(pooled_hidden_states)
                    processed_files.append(verilog_file_path)

                    success = True  # Mark success if all splits were processed without errors
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        split_factor *= 2  # Increase split factor to reduce memory usage
                        torch.cuda.empty_cache()  # Clear memory to try again
                    else:
                        raise e

            if count % 10 == 0:
                save_module_embeddings(processed_files, pooled_hidden_states_list, embeddings_save_path)
            count += 1

        except Exception as e:
            print(f"Error processing file {verilog_file_path}: {e}")
        
        progress_bar.update(1)

    save_module_embeddings(processed_files, pooled_hidden_states_list, embeddings_save_path)

def get_line_embeddings(model, tokenizer, pooling_fn, relative_verilog_paths, preprocessors):

    filtered_verilog_paths = []

    for verilog_file_path in relative_verilog_paths:
        embeddings_file_path = verilog_file_path.replace('openROAD_low_level_modules_yosys_v1', '../embeddings/openROAD_low_level_modules_yosys_v1_embeddings')
        embeddings_file_path = embeddings_file_path.replace('.v', '_embeddings.pkl')

        if not os.path.exists(embeddings_file_path): 
            filtered_verilog_paths.append(verilog_file_path)

    relative_verilog_paths = filtered_verilog_paths

    num_training_steps = len(relative_verilog_paths)
    progress_bar = tqdm(range(num_training_steps))

    for verilog_file_path in relative_verilog_paths:
        embeddings_file_path = verilog_file_path.replace('openROAD_low_level_modules_yosys_v1', '../embeddings/openROAD_low_level_modules_yosys_v1_embeddings')
        embeddings_file_path = embeddings_file_path.replace('.v', '_embeddings.pkl')

        parent_dir = os.path.dirname(embeddings_file_path)
        os.makedirs(parent_dir, exist_ok=True)

        if os.path.exists(embeddings_file_path): 
            progress_bar.update(1)
            continue
        
        torch.cuda.empty_cache()

        try:
            verilog_code = read_file(verilog_file_path)
            # preprocessed_verilog_code = preprocess_code(verilog_code, preprocessors)
            verilog_code_lines = verilog_code.split('\n')
            verilog_code_lines = get_context_lines(verilog_code_lines)
            batched_lines, batch_indices = batch_with_indices(verilog_code_lines)
            embeddings = {}

            for batch_code, batch_index in zip(batched_lines, batch_indices):
                inputs = tokenizer(batch_code, return_tensors="pt", padding=True).to('cuda')
                attention_mask = inputs['attention_mask'].to('cuda')
                input_ids = inputs['input_ids'].to('cuda')
                # print(attention_mask)
        
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]
        
                for i, line_no in enumerate(batch_index):
                    verilog_code_line = verilog_code_lines[line_no]
                    hidden_state = hidden_states[i]
                    attn_mask = attention_mask[i]
                    # Get sum_hidden_states and valid_token_count for the split
                    sum_hidden_states, valid_token_count = pooling_fn(hidden_state, attn_mask)
        
                    # Compute overall pooled hidden states by dividing cumulative sum by cumulative valid token count
                    valid_token_count = valid_token_count.clamp(min=1e-9)
                    pooled_hidden_states = sum_hidden_states / valid_token_count
                    embeddings[line_no] = {"line": verilog_code_line, "embedding": pooled_hidden_states.to('cpu').detach().numpy(), "token_count": valid_token_count.to('cpu').detach().numpy()}
                    
                del inputs, attention_mask, input_ids, outputs, hidden_states

            with open(embeddings_file_path, 'wb') as f:
                pickle.dump(embeddings, f)

        except Exception as e:
            print(f"Error processing file {verilog_file_path}: {e}")
        
        progress_bar.update(1)

def main():
    parser = argparse.ArgumentParser(description='Get embedding type.')
    parser.add_argument('--embedding_type', type=str, choices=['module', 'line'], required=True, help='Type of embedding: module or line')
    args = parser.parse_args()

    embedding_type = args.embedding_type

    csv_data_file = "../data/processed/RTL_dataset/dump/synthesis_data_nangate45nm.csv"

    relative_verilog_paths = get_relative_verilog_paths_from_csv(csv_data_file)
    relative_verilog_paths = list(set(relative_verilog_paths))
    preprocessors = [remove_comments, remove_multiple_newlines]

    model, tokenizer = load_cl_verilog_model()

    if embedding_type == 'module':
        pooling_fn = module_attention_mask_based_pooling
        get_module_embeddings(model, tokenizer, pooling_fn, relative_verilog_paths, preprocessors)
    elif embedding_type == 'line':
        pooling_fn = line_attention_mask_based_pooling
        get_line_embeddings(model, tokenizer, pooling_fn, relative_verilog_paths, preprocessors)
    
if __name__ == '__main__':
    main()