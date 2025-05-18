import sys


# sys.path.append('/home/csp/repo/LLMs/eye_transformer/')
# print("CWD", os.getcwd(), "PATH", sys.path)
from utils.lmdb_storage import LMDBStorage
import pathlib
import hashlib

sys.path.append("../..")
import numpy as np

path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from eyetrackpy.data_generator.fixations_predictor_trained_1.fixations_predictor_model_1 import (
    FixationsPredictor_1,
)

from eyetrackpy.data_generator.fixations_predictor_trained_2.fixations_predictor_model_2 import (
    FixationsPredictor_2,
)
from fixations_predictor_model_6 import FixationsPredictor_6
from typing import (
    TypeVar,
)

T = TypeVar("T", bound="Module")
import re


class MyRewardBase:
    def __init__(
        self,
        model_name,
        features_used=[1, 1, 1, 1, 1],
        *argv,
        **karg,
    ):
        self.features_used = features_used
        self.model_name = model_name
        self.memory_storage = LMDBStorage(
            db_path="buffer_train.lmdb"
        )

    def _load_tokenizer(self, load_local_folder_name=None):
        LLAMA_TEMPLATE = """{% for message in messages %}
            {% if message['role'] == 'user' %}
                {{ bos_token + ' ' + message['content'] + ' ' }}
            {% elif message['role'] == 'system' %}
                {{ '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n\n' }}
            {% elif message['role'] == 'assistant' %}
                {{ ' ' + message['content'] + ' ' + eos_token }}
            {% endif %}
        {% endfor %}"""
        if load_local_folder_name:
            tokenizer = AutoTokenizer.from_pretrained(load_local_folder_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        if tokenizer.chat_template is None:
            #tokenizer.chat_template = tokenizer.default_chat_template
            tokenizer.chat_template = LLAMA_TEMPLATE
        tokenizer.add_eos_token = True
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.padding_side = "right"
        chat_tokens = list(set(re.findall(r"(<.*?>)", tokenizer.get_chat_template())))

        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": tokenizer.additional_special_tokens
                + chat_tokens
            }
        )
        self.tokenizer = tokenizer
        return self.tokenizer

    def load_fx_model_1(self, hidden_size, remap=False, fp_dropout=[0.0, 0.3]):
        p_1, p_2 = fp_dropout

        self.modelTokenizer = self.tokenizer
        self.FP_model = FixationsPredictor_1(
            hidden_dim=128,
            drop_out=0.2,
            modelTokenizer=self.modelTokenizer,
            remap=remap,
        )
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(1, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def load_fx_model_2(
        self,
        hidden_size,
        remap=False,
        fp_dropout=[0.0, 0.3],
        load_fix_model=True,
    ):
        p_1, p_2 = fp_dropout
        self.modelTokenizer = self.tokenizer
        if load_fix_model:
            self.FP_model = FixationsPredictor_2(
                modelTokenizer=self.modelTokenizer, remap=remap
            )
        num_features = int(sum(self.features_used))
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(num_features, 128),
            # nn.BatchNorm1d(128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)
    def load_fx_model_4(
            self,
            hidden_size,
            remap=False,
            fp_dropout=[0.0, 0.3],
    ):
        p_1, p_2 = fp_dropout
        self.modelTokenizer = self.tokenizer
        # No FixationsPredictor model is loaded for random embeddings
        num_features = int(sum(self.features_used))
        # Projector remains to shape random features into hidden_size
        self.fixations_embedding_projector = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=p_1),
            nn.Linear(128, hidden_size),
            nn.Dropout(p=p_2),
        )
        self.norm_layer_fix = nn.LayerNorm(hidden_size)

    def load_fx_model_6(
        self,
        hidden_size,
        remap=False,
        fp_dropout=[0.0, 0.3],
        model_paths=None,
        num_models=None,
    ):
        from fixations_predictor_model_6 import (
            FixationsPredictor_6,
        )

        self.fixations_predictor = FixationsPredictor_6(
            modelTokenizer=self.tokenizer,
            remap=remap,
            model_paths=model_paths,
            num_models=num_models,
        )
        self.fixations_predictor.to(self.model.device)
        self.fixations_predictor.eval()  # Set to evaluation mode since we're only doing inference

    def get_text_sample_from_lmdb(self, hash_id):
        """Get a text sample from LMDB storage using hash_id."""
        result = self.memory_storage.getItem(hash_id)
        if result is None:
            return None
        return result.get("text", None)

    def _compute_fixations(
        self, input_ids, attention_mask, remap=False, fixations_model_version=1
    ):
        if fixations_model_version == 1:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids)
        elif fixations_model_version == 2:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids, attention_mask)
        elif fixations_model_version == 5:
            batch_size, seq_len = input_ids.size()
            
            # Calculate number of tokens to replace and shuffle (vectorized)
            replace_ratio = torch.rand(batch_size, device=input_ids.device) * 0.2 + 0.4  # [batch_size]
            shuffle_ratio = torch.rand(batch_size, device=input_ids.device) * 0.1 + 0.2  # [batch_size]
            num_tokens_to_replace = (seq_len * replace_ratio).long()  # [batch_size]
            num_tokens_to_shuffle = (seq_len * shuffle_ratio).long()  # [batch_size]
            
            # Create a copy of input_ids to modify
            modified_input_ids = input_ids.clone()
            
            # Create mask for valid positions (excluding special tokens)
            valid_positions = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
            valid_positions[:, 0] = False  # Skip first token
            valid_positions[:, -1] = False  # Skip last token
            
            # Get token embeddings for similarity comparison
            with torch.no_grad():
                token_embeddings = self.model.embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]
                vocab_embeddings = self.model.embed_tokens.weight  # [vocab_size, hidden_size]
                vocab_size = vocab_embeddings.size(0)
                
                # Process vocabulary in chunks to avoid memory issues
                vocab_chunk_size = 100  # Increased chunk size for better efficiency
                num_vocab_chunks = (vocab_size + vocab_chunk_size - 1) // vocab_chunk_size
                
                # For each sequence in the batch
                for b in range(batch_size):
                    # Get valid indices for this sequence
                    valid_indices = torch.where(valid_positions[b])[0]
                    
                    # Randomly select tokens to replace
                    replace_indices = valid_indices[torch.randperm(len(valid_indices), device=input_ids.device)[:num_tokens_to_replace[b]]]
                    
                    # Get embeddings for tokens to replace
                    tokens_to_replace = token_embeddings[b, replace_indices]  # [num_tokens, hidden_size]
                    
                    # Initialize similarity scores
                    best_similarities = torch.zeros(len(replace_indices), device=input_ids.device)
                    best_token_ids = torch.zeros(len(replace_indices), dtype=torch.long, device=input_ids.device)
                    
                    # Process vocabulary in chunks
                    for v_chunk_idx in range(num_vocab_chunks):
                        v_start = v_chunk_idx * vocab_chunk_size
                        v_end = min((v_chunk_idx + 1) * vocab_chunk_size, vocab_size)
                        vocab_chunk = vocab_embeddings[v_start:v_end]  # [chunk_size, hidden_size]
                        
                        # Compute similarities for current vocabulary chunk
                        chunk_similarities = torch.nn.functional.cosine_similarity(
                            tokens_to_replace.unsqueeze(1),  # [num_tokens, 1, hidden_size]
                            vocab_chunk.unsqueeze(0),        # [1, chunk_size, hidden_size]
                            dim=2
                        )  # [num_tokens, chunk_size]
                        
                        # Update best similarities and token IDs
                        chunk_max_similarities, chunk_max_indices = torch.max(chunk_similarities, dim=1)
                        better_mask = chunk_max_similarities > best_similarities
                        best_similarities[better_mask] = chunk_max_similarities[better_mask]
                        best_token_ids[better_mask] = chunk_max_indices[better_mask] + v_start
                    
                    # Get top k similar tokens and randomly select one
                    top_k = 20
                    top_k_values, top_k_indices = torch.topk(best_similarities, min(top_k, len(replace_indices)), dim=0)
                    random_selections = torch.randint(0, min(top_k, len(replace_indices)), (len(replace_indices),), device=input_ids.device)
                    new_token_ids = best_token_ids[top_k_indices[random_selections]]
                    
                    # Replace the tokens
                    modified_input_ids[b, replace_indices] = new_token_ids
                    
                    # Handle token shuffling
                    remaining_valid_indices = torch.tensor([i for i in valid_indices if i not in replace_indices], device=input_ids.device)
                    
                    if len(remaining_valid_indices) > 0:
                        # Randomly select tokens to shuffle
                        shuffle_indices = remaining_valid_indices[torch.randperm(len(remaining_valid_indices), device=input_ids.device)[:num_tokens_to_shuffle[b]]]
                        
                        if len(shuffle_indices) > 1:
                            # Get and shuffle the tokens
                            tokens_to_shuffle = modified_input_ids[b, shuffle_indices]
                            shuffled_tokens = tokens_to_shuffle[torch.randperm(len(tokens_to_shuffle), device=input_ids.device)]
                            modified_input_ids[b, shuffle_indices] = shuffled_tokens
                    
                    # Clear cache after each sequence
                    torch.cuda.empty_cache()
            # Use the modified input_ids to compute fixations
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(modified_input_ids, attention_mask)
        elif fixations_model_version == 4:
            # Generate random embeddings with shape (batch_size, seq_len, num_features)
            batch_size, seq_len = input_ids.size()
            num_features = sum(self.features_used)
            fixations = torch.rand(
                batch_size, seq_len, 5,
                device=input_ids.device, dtype=torch.float
            )
            fixations_attention_mask = torch.ones(
                (batch_size, seq_len),
                device=input_ids.device,
                dtype=attention_mask.dtype
            )
            return (
                fixations,
                fixations_attention_mask,
                None,  # mapped_fixations (unused)
                None,  # text_tokenized_model (unused)
                None,  # text_tokenized_fix (unused)
                None,  # sentences (unused)
            )
        elif fixations_model_version == 6:
            (
                fixations,
                fixations_attention_mask,
                mapped_fixations,
                text_tokenized_model,
                text_tokenized_fix,
                sentences,
            ) = self.FP_model._compute_mapped_fixations(input_ids, attention_mask)
        if remap:
            fixations_attention_mask = attention_mask
        return (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        )

    def compute_fixations(
        self, input_ids, attention_mask, remap=False, fixations_model_version=1
    ):
        # compute fixations you can use the cached funcion that will try to search from them on the cache.
        # there are not identical, because when computing you are using batch size, but when computing to save not
        # (
        #     fixations2,
        #     fixations_attention_mask2,
        #     mapped_fixations2,
        #     text_tokenized_model2,
        #     text_tokenized_fix2,
        #     sentences2,
        # ) = self._compute_fixations(
        #     input_ids, attention_mask, remap, fixations_model_version
        # )
        (
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            text_tokenized_model,
            text_tokenized_fix,
            sentences,
        ) = self.compute_fixations_cached(
            input_ids, attention_mask, remap, fixations_model_version
        )
        del text_tokenized_fix, text_tokenized_model, sentences
        fixations_normalized, fixations_attention_mask = self.process_fixations(
            fixations,
            fixations_attention_mask,
            mapped_fixations,
            remap,
            fixations_model_version,
        )
        return fixations_normalized, fixations_attention_mask

    def process_fixations(
        self,
        fixations,
        fixations_attention_mask,
        mapped_fixations,
        remap,
        fixations_model_version,
    ):
        # compute fixations
        if remap:
            fixations = mapped_fixations
            # mapped_fixations = mapped_fixations.detach()
            del mapped_fixations
        # add noise compute fixations
        if self.training is False and self.noise_factor > 0:
            noise = torch.randn_like(fixations) * self.noise_factor
            fixations = fixations + noise
            noise = noise.detach()
            del noise
            
        # Ensure fixations have the correct shape for the projector
        if fixations_model_version in [5, 6]:
            # For versions 5 and 6, we need to ensure the fixations have the correct shape
            if fixations.dim() == 1:  # If it's 1D, reshape to 2D
                fixations = fixations.unsqueeze(-1)  # Add feature dimension
            elif fixations.dim() == 3:  # [batch, seq_len, features]
                # Select only the features we want to use
                idx = np.where(np.array(self.features_used) == 1)[0]
                if len(idx) == 0:
                    raise ValueError("No features selected in features_used")
                fixations = fixations[:, :, idx]
                
                # Reshape to [batch * seq_len, features] for the linear layer
                batch_size, seq_len, num_features = fixations.shape
                fixations = fixations.reshape(-1, num_features)
            
        # project to embedding size dimension
        fixations_projected = self.fixations_embedding_projector(fixations)
        
        # normalize fixations
        fixations_normalized = self.norm_layer_fix(fixations_projected)
        
        # Reshape back to [batch, seq_len, hidden_size] if needed
        if fixations_model_version in [5, 6]:
            fixations_normalized = fixations_normalized.reshape(batch_size, seq_len, -1)
            
        # Ensure attention mask stays 2D [batch, seq_len]
        if fixations_attention_mask.dim() > 2:
            fixations_attention_mask = fixations_attention_mask[:, :, 0]  # Take first feature dimension
            
        torch.cuda.empty_cache()
        return fixations_normalized, fixations_attention_mask

    @staticmethod
    def hash_value(val):
        return hashlib.md5(str(val).encode()).hexdigest()

    @staticmethod
    def remove_padding_from_batch(batch_token_ids, pad_token_id=0):
        # Iterate over each sequence in the batch and remove padding
        return [
            list(filter(lambda token_id: token_id != pad_token_id, sequence))
            for sequence in batch_token_ids
        ]

    def compute_fixations_cached(
        self, input_ids_original, attention_mask, remap=False, fixations_model_version=1
    ):
        device = input_ids_original.device
        # Convert the input tensor to a list of lists and remove padding.
        input_ids_list = input_ids_original.cpu().numpy().tolist()
        filtered_ids = self.remove_padding_from_batch(
            input_ids_list, self.tokenizer.pad_token_id
        )
        fixations_all, fixations_attention_mask_all = [], []
        # Iterate over each sequence in the filtered list
        for seq in filtered_ids:
            # Compute a hash of the sequence for caching
            if remap is True:
                # we only care about the model if we are remapping

                hash_id = self.hash_value(
                    seq + [fixations_model_version] + ["remap"] + [self.model_name]
                )
            else:
                hash_id = self.hash_value(
                    seq + [fixations_model_version] + [self.model_name]
                )  # because we can call the same sequence with and without remap and diff fixations predictor model

            # Attempt to retrieve the result from the cache
            result = self.memory_storage.getItem(hash_id)

            if result is None:
                # If the result is not in the cache, compute the fixations
                print("result NOT found")
                torch_seq = torch.LongTensor(np.asarray(seq)).to(device).unsqueeze(0)
                (
                    fixations,
                    fixations_attention_mask,
                    mapped_fixations,
                    text_tokenized_model,
                    text_tokenized_fix,
                    sentences,
                ) = self._compute_fixations(
                    torch_seq,
                    attention_mask,
                    remap=remap,
                    fixations_model_version=fixations_model_version,
                )
                del text_tokenized_fix, text_tokenized_model, sentences
                if remap:
                    fixations = mapped_fixations
                    fixations_attention_mask = attention_mask
                fixation_outputs = {
                    "fixations": fixations.cpu(),
                    "fixations_attention_mask": fixations_attention_mask.cpu(),
                }
                self.memory_storage.add(hash_id, fixation_outputs)
            else:
                print("result found")
                # If the result is found in the cache, convert back to tensors
                fixations = result["fixations"].to(device)
                fixations_attention_mask = result["fixations_attention_mask"].to(device)
            if fixations_model_version == 2 or fixations_model_version == 4:
                idx = np.where(np.array(self.features_used) == 1)[0]
                fixations = fixations[:, :, idx]
            fixations_all.append(fixations.squeeze())
            fixations_attention_mask_all.append(fixations_attention_mask.squeeze())

        # ---------------
        # Pad and concatenate all outputs into the final result tensor
        fixations_all = self._pad_and_concat(fixations_all)
        if remap is False:
            fixations_attention_mask_all = self._pad_and_concat(
                fixations_attention_mask_all
            )
            return fixations_all, fixations_attention_mask_all, None, None, None, None
        else:
            try:
                fixations_attention_mask_all = self._pad_and_concat(
                    fixations_attention_mask_all
                )
            except:
                # enter here on the last of the batch, take a look
                print(
                    f"problema con el remapping con len {len(fixations_attention_mask_all)}, {len(fixations_attention_mask_all[0])}"
                )
                print(fixations_attention_mask_all)

            fixations_attention_mask_all = attention_mask
            return None, fixations_attention_mask_all, fixations_all, None, None, None

    @staticmethod
    def _pad_and_concat(list_of_tensors):
        def pad_tensor(tensor, max_length):
            """Pads a tensor to the specified max_length with the last value in the tensor."""
            # If tensor has extra dimensions, squeeze them
            if tensor.dim() > 2:
                tensor = tensor.squeeze()
            elif tensor.dim() == 2:
                # If it's a 2D tensor, check if it's a batch dimension
                if tensor.size(0) == 1:
                    tensor = tensor.squeeze(0)
                elif tensor.size(1) == 1:
                    tensor = tensor.squeeze(1)
                else:
                    # If it's a genuine 2D tensor, keep it as is
                    pass
            
            padding_length = max_length - tensor.size(0)
            if padding_length > 0:
                if tensor.dim() == 1:  # 1D tensor
                    padding_tensor = torch.zeros(padding_length).to(tensor.device)
                elif tensor.dim() == 2:  # 2D tensor
                    padding_tensor = torch.zeros(padding_length, tensor.size(1)).to(tensor.device)
                else:
                    raise ValueError("Only 1D and 2D tensors are supported.")
                tensor = torch.cat([tensor, padding_tensor])
            return tensor

        # Determine the maximum size for padding for each tensor position
        max_length = max([len(i) for i in list_of_tensors])

        # Check if we have a mix of 1D and 2D tensors
        has_2d = any(t.dim() == 2 for t in list_of_tensors)
        has_1d = any(t.dim() == 1 for t in list_of_tensors)
        
        if has_2d and has_1d:
            # Convert all tensors to 2D
            list_of_tensors = [t.unsqueeze(-1) if t.dim() == 1 else t for t in list_of_tensors]

        result = torch.stack([pad_tensor(item, max_length) for item in list_of_tensors])
        return result
