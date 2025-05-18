import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import os
from typing import List

class FixationsPredictor_6(nn.Module):
    def __init__(self, modelTokenizer=None, remap=False, model_paths=None, num_models=None):
        super(FixationsPredictor_6, self).__init__()
        self.remap = remap
        self.modelTokenizer = modelTokenizer
        
        # Load ensemble of roberta-large models
        if model_paths is None:
            raise ValueError("model_paths must be provided to load the trained roberta-large models")
        
        # If model_paths is a directory, get all .pt files
        if os.path.isdir(model_paths):
            all_paths = [os.path.join(model_paths, f) for f in os.listdir(model_paths) 
                        if f.endswith('.pt')]
            if num_models is not None:
                all_paths = all_paths[:num_models]
            if not all_paths:
                raise ValueError(f"No .pt model files found in {model_paths}")
        else:
            all_paths = [model_paths]
            
        print(f"Loading {len(all_paths)} roberta-large models for ensemble")
        
        # Initialize list of models and decoders
        self.roberta_models = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for path in all_paths:
            if not os.path.exists(path):
                raise ValueError(f"Model path {path} does not exist")
                
            # Load base model
            model = RobertaModel.from_pretrained('roberta-large')
            decoder = nn.Sequential(nn.Linear(1024, 5))
            
            # Load state dict from .pt file
            checkpoint = torch.load(path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Split state dict into roberta and decoder parts
            roberta_state_dict = {}
            decoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('roberta.'):
                    roberta_state_dict[k[8:]] = v  # Remove 'roberta.' prefix
                elif k.startswith('decoder.'):
                    decoder_state_dict[k[8:]] = v  # Remove 'decoder.' prefix
                else:
                    decoder_state_dict[k] = v  # Assume remaining keys are for decoder
            
            # Load state dicts
            model.load_state_dict(roberta_state_dict)
            decoder.load_state_dict(decoder_state_dict)
            
            model.eval()  # Set to evaluation mode since we're only doing inference
            decoder.eval()
            
            self.roberta_models.append(model)
            self.decoders.append(decoder)

    def _compute_mapped_fixations(self, input_ids, attention_mask=None):
        # Ensure input_ids and attention_mask have the same batch size
        if attention_mask is not None and input_ids.size(0) != attention_mask.size(0):
            # Take the first batch from attention_mask if it's larger
            if attention_mask.size(0) > input_ids.size(0):
                attention_mask = attention_mask[:input_ids.size(0)]
            # Expand input_ids if it's smaller
            elif input_ids.size(0) < attention_mask.size(0):
                input_ids = input_ids.expand(attention_mask.size(0), -1)

        # Get batch size from input_ids
        batch_size = input_ids.size(0)
        
        # Process each model in the ensemble
        all_predictions = []
        for i, model in enumerate(self.roberta_models):
            # Move tensors to the same device as the model
            input_ids = input_ids.to(model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            # Ensure input_ids has batch dimension
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            # Truncate to model's maximum sequence length (512 for RoBERTa)
            max_seq_length = 512
            if input_ids.size(1) > max_seq_length:
                input_ids = input_ids[:, :max_seq_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :input_ids.size(1)]
            
            # Forward pass through the model
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Get hidden states and predictions
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            predictions = self.decoders[i](hidden_states)
            
            # Ensure predictions have the same sequence length as input_ids
            if predictions.size(1) != input_ids.size(1):
                predictions = predictions[:, :input_ids.size(1), :]
            
            all_predictions.append(predictions)
        
        # Average predictions from all models
        fixations = torch.stack(all_predictions).mean(dim=0)  # [batch_size, seq_len, num_features]
        
        # Create attention mask for fixations if not provided
        if attention_mask is None:
            fixations_attention_mask = torch.ones_like(input_ids, dtype=torch.float)
        else:
            fixations_attention_mask = attention_mask.clone()
        
        # Ensure attention mask has the same shape as fixations
        if fixations_attention_mask.dim() == 2:
            fixations_attention_mask = fixations_attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Ensure attention mask has the same sequence length as fixations
        if fixations_attention_mask.size(1) != fixations.size(1):
            fixations_attention_mask = fixations_attention_mask[:, :fixations.size(1), :]
        
        # Ensure attention mask has the same batch size as fixations
        if fixations_attention_mask.size(0) != fixations.size(0):
            fixations_attention_mask = fixations_attention_mask[:fixations.size(0)]
        
        return (
            fixations,
            fixations_attention_mask,
            fixations,  # mapped_fixations
            input_ids,  # text_tokenized_model
            input_ids,  # text_tokenized_fix
            None,  # sentences
        )

    def forward(self, input_ids, attention_mask=None):
        fixations, fixations_attention_mask, _, _, _, _ = self._compute_mapped_fixations(
            input_ids, attention_mask
        )
        return fixations, fixations_attention_mask