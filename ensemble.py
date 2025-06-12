# ensemble.py

import torch
import math
import os
from typing import List, Optional, Set, Dict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GenerationAsClassification:
    """
    Handles the ensemble model loading, preparation, and generation logic.
    Uses a UNION vocabulary approach with UNK handling.
    """

    def __init__(self, config_dict: Dict):
        self.config = config_dict
        
        # Extract configuration values with defaults
        self.model_names = config_dict["model_names"]
        self.tokenizer_names = config_dict["tokenizer_names"] 
        self.device_config = config_dict.get("device", "cpu")
        self.devices_config = config_dict.get("devices", None)
        self.model_type = config_dict.get("model_type", None)
        self.weights_config = config_dict.get("weights", None)
        self.token = config_dict.get("token", None)
        self.model_count = config_dict.get("model_count", None)
        self.trust_remote_code = config_dict.get("trust_remote_code", True)
        # Check and print CUDA devices
        print("[CUDA] Running CUDA device check...")
        cuda_count = torch.cuda.device_count()
        print(f"[CUDA] PyTorch sees {cuda_count} CUDA devices")
        
        if cuda_count == 0:
            print("[CUDA] No CUDA devices available! Using CPU.")
            self.devices = [torch.device("cpu")]
            self.device = self.devices[0]
        elif cuda_count == 1:
            print("[CUDA] Only one CUDA device available. All models will use the same GPU.")
            self.devices = [torch.device("cuda:0")]
            self.device = self.devices[0]
        else:
            print(f"[CUDA] Multiple CUDA devices available: {cuda_count}")
            # Force use of separate devices for each model if possible
            if len(self.model_names) > 1:
                num_models = len(self.model_names)
                num_devices = min(num_models, cuda_count)
                self.devices = [torch.device(f"cuda:{i % cuda_count}") for i in range(num_devices)]
                
                # Extend devices list if there are more models than GPUs
                while len(self.devices) < num_models:
                    self.devices.append(self.devices[len(self.devices) % cuda_count])
                    
                print(f"[CUDA] Forced device assignment: {self.devices}")
            else:
                # Use devices from config
                if self.devices_config:
                    self.devices = []
                    for d in self.devices_config:
                        if d.startswith('cuda:'):
                            idx = int(d.split(':')[1])
                            if idx < cuda_count:
                                self.devices.append(torch.device(d))
                            else:
                                print(f"[CUDA] Device {d} index exceeds available count, using cuda:0")
                                self.devices.append(torch.device("cuda:0"))
                        else:
                            self.devices.append(torch.device(d))
                else:
                    self.devices = [torch.device("cuda:0")]
                    
            self.device = self.devices[0]
        
        print(f"[CUDA] Final device setup: {self.devices}")
        
        # Initialize class-level EOS token attributes
        self.eos_token_strings = set()
        self.eos_token_indices_in_union = set()

        print("[GAC __init__] Initializing GAC Models...")
        self.models = []
        self.tokenizers = []
        
        # Load models
        self._load_models()
        
        # Load tokenizers
        self._load_tokenizers()

        self.weights = torch.tensor(
            (
                self.config["weights"]
                if self.config.get("weights")
                else [1.0 / len(self.models)] * len(self.models)
            ),
            device=self.device,
        )
        
        # Initialize vocab/mapping attributes
        self.union_vocab_set = None
        self.union_vocab_list = None
        self.union_vocab_token_to_idx = None
        self.union_vocab_idx_to_token = None
        self.mappings = []
        
        # Initialize EOS attributes
        self.primary_eos_token = None
        self.primary_eos_idx_union = None

        print("[GAC __init__] GAC Initialized.")

    def _load_models(self):
        """Load models and distribute across available devices."""
        for idx, name in enumerate(self.config["model_names"]):
            # Select device for this model using round-robin
            device_idx = idx % len(self.devices)
            device = self.devices[device_idx]
            
            print(f"  Loading model #{idx}: {name} on {device}")
            
            try:
                # Check CUDA availability for this device
                if device.type == 'cuda':
                    # Verify the device is actually available
                    cuda_count = torch.cuda.device_count()
                    print(f"    CUDA device count: {cuda_count}")
                    
                    if device.index >= cuda_count:
                        print(f"    WARNING: Device {device} index exceeds available count, using cuda:0 instead")
                        device = torch.device('cuda:0')
                    
                    # For safety, initialize the device before loading
                    with torch.cuda.device(device):
                        # Just allocate a small tensor to ensure the device is initialized
                        torch.zeros(1, device=device)
                        print(f"    Device {device} initialized successfully")
                
                model = AutoModelForCausalLM.from_pretrained(
                    name,
                    use_auth_token=self.config.get("token", None),
                    trust_remote_code=True,
                    torch_dtype=torch.float16,   
                    low_cpu_mem_usage=True
                ).to(device)  # Use .to() after loading
                
                self.models.append(model)
                
                print(f"    Model {name} loaded on {device}.")
            except Exception as e:
                print(f"    ERROR loading model {name}: {e}")
                import traceback
                print(f"    Full traceback: {traceback.format_exc()}")
                

    def _load_tokenizers(self):
        """Load tokenizers for all models."""
        print("[GAC __init__] Initializing GAC Tokenizers...")
        for name in self.config["tokenizer_names"]:
            print(f"  Loading tokenizer: {name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    name, use_auth_token=self.config.get("token", None), trust_remote_code=True
                )
                
                tokenizer.padding_side = "left"
                self.tokenizers.append(tokenizer)
            except Exception as e:
                print(f"    ERROR loading tokenizer {name}: {e}")
                import traceback
                print(f"    Full traceback: {traceback.format_exc()}")
        
        print(f"[DEBUG] Loaded {len(self.tokenizers)} tokenizers")
        print("[GAC __init__] Tokenizers loaded.")

    def _create_vocab(self):
        """Create a simple UNION vocabulary without tokenizer-specific normalization."""
        print("[GAC _create_vocab] Creating universal UNION vocabulary...")
        
        # Initialize for the union vocabulary
        self.union_vocab_set = set()
        
        # Simply collect all unique tokens from all tokenizers
        for i, tokenizer in enumerate(self.tokenizers):
            try:
                vocab = tokenizer.get_vocab()
                self.union_vocab_set.update(vocab.keys())
                print(f"  Added {len(vocab)} tokens from tokenizer {i}")
            except Exception as e:
                print(f"  Error processing tokenizer {i}: {e}")
        
        # Create sorted list and mappings
        self.union_vocab_list = sorted(list(self.union_vocab_set))
        self.union_vocab_token_to_idx = {
            token: idx for idx, token in enumerate(self.union_vocab_list)
        }
        self.union_vocab_idx_to_token = {
            idx: token for token, idx in self.union_vocab_token_to_idx.items()
        }
        
        print(f"  Union vocabulary size: {len(self.union_vocab_list)}")
        if not self.union_vocab_list:
            print("[ERROR] Union vocabulary is empty!")

    def _create_mappings(self):
        """Create simple mappings from each model's vocabulary to the UNION vocabulary indices."""
        print("[GAC _create_mappings] Creating universal mappings to UNION vocabulary...")
        self.mappings = []
        
        if self.union_vocab_token_to_idx is None:
            raise RuntimeError("Union vocabulary must be created before mappings.")
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            try:
                model_vocab_size = model.config.vocab_size
                print(f"  Mapping for model {i} (Vocab size: {model_vocab_size}) -> Union Size {len(self.union_vocab_list)}")
                
                # Create mapping array on CPU first
                mapping = torch.full((model_vocab_size,), -1, dtype=torch.long, device="cpu")
                
                # Get tokenizer vocabulary
                vocab = tokenizer.get_vocab()
                mapped_count = 0
                
                # Simple direct mapping - no normalization
                for token, token_id in vocab.items():
                    if token_id >= model_vocab_size:
                        continue  # Skip tokens with IDs outside the model's vocab size
                    
                    # Direct lookup in union vocabulary
                    if token in self.union_vocab_token_to_idx:
                        union_idx = self.union_vocab_token_to_idx[token]
                        mapping[token_id] = union_idx
                        mapped_count += 1
                    # If token not found, leave as -1 (will be treated as UNK)
                
                print(f"    Mapped {mapped_count}/{len(vocab)} tokens directly")
                
                # Move the mapping to the device where the model is
                model_device = model.device
                mapping_device = mapping.to(model_device)
                self.mappings.append(mapping_device)
                
            except Exception as e:
                print(f"    [ERROR] Mapping creation exception: {e}")
                raise RuntimeError(f"Mapping creation failed for model {i}") from e
        
        print("[GAC _create_mappings] Mappings created successfully.")

    def prepare(self, models=None, tokenizers=None, devices=None):
        """Prepare the strategy by creating vocabularies and mappings."""
        print("[GAC prepare] Preparing GAC strategy (Union Vocab)...")
        
        # Use provided models/tokenizers or the ones already loaded
        if models is not None:
            self.models = models
        if tokenizers is not None:
            self.tokenizers = tokenizers
        if devices is not None:
            self.devices = devices if isinstance(devices, list) else [devices]
            
        if not self.models or not self.tokenizers:
            raise ValueError("Need models/tokenizers")
            
        self._create_vocab()
        print("  Union vocabulary created.")
        
        self._create_mappings()
        print("  Mappings created.")
        
        # Setup EOS tokens after vocab and mappings are created
        self._setup_eos_tokens()
        print("[GAC prepare] Strategy preparation complete.")

    def _setup_eos_tokens(self):
        """Setup EOS tokens and indices at the class level"""
        self.eos_token_strings = set()
        self.eos_token_indices_in_union = set()
        potential_eos = ["<|endoftext|>", "<|eot_id|>", "</s>", "<|im_end|>"]

        # Process on CPU for safety
        for t in self.tokenizers:
            if t.eos_token:
                self.eos_token_strings.add(t.eos_token)
                idx = self.union_vocab_token_to_idx.get(t.eos_token)
                if idx is not None:
                    self.eos_token_indices_in_union.add(idx)
            for ts in potential_eos:
                if ts in t.get_vocab():
                    self.eos_token_strings.add(ts)
                    idx = self.union_vocab_token_to_idx.get(ts)
                    if idx is not None:
                        self.eos_token_indices_in_union.add(idx)

        self.primary_eos_token = (
            self.tokenizers[0].eos_token if self.tokenizers else None
        )
        self.primary_eos_idx_union = (
            self.union_vocab_token_to_idx.get(self.primary_eos_token)
            if self.primary_eos_token
            else None
        )

        if self.primary_eos_idx_union is None and self.eos_token_indices_in_union:
            self.primary_eos_idx_union = next(iter(self.eos_token_indices_in_union))

        if not self.eos_token_indices_in_union:
            print("[WARN] _setup_eos_tokens: No EOS tokens found in union vocab.")
        else:
            print(f"[INFO] Primary EOS token: '{self.primary_eos_token}' (Union Idx: {self.primary_eos_idx_union})")
            print(f"[INFO] All EOS tokens in union: {self.eos_token_strings}")

    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        top_p: float = 0.9,
        temperature: float = 0.7,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ):
        """Generate text using the ensemble with GAC's approach, with improved device handling."""
        if self.mappings is None or self.union_vocab_idx_to_token is None:
            raise RuntimeError("Model not prepared.")
            
        if stop_sequences is None:
            stop_sequences = ["\n", ".\n", "?\n", "!\n", "\n\n"]
        
        # Start with prompt
        generated_text = prompt
        generated_tokens_count = 0
        
        # Encode prompt for each model, placing input tensors on the appropriate device
        input_ids_list = []
        for i, tokenizer in enumerate(self.tokenizers):
            # Get the device for this model
            model_device = self.models[i].device
            # Encode and place on the correct device
            ids = tokenizer.encode(prompt, return_tensors="pt").to(model_device)
            input_ids_list.append(ids)
        
        # Generation loop
        while generated_tokens_count < max_length:
            # Get logits and convert to probabilities
            with torch.no_grad():
                model_probs = []
                valid_indices = []
                
                for i, (model, ids) in enumerate(zip(self.models, input_ids_list)):
                    try:
                        # Ensure input is on model's device
                        model_device = model.device
                        if ids.device != model_device:
                            ids = ids.to(model_device)
                        
                        # Handle maximum sequence length
                        max_len = getattr(model.config, 'max_position_embeddings', 2048)
                        if ids.shape[1] > max_len:
                            ids = ids[:, -max_len:]
                        
                        # Forward pass
                        logits = model(ids).logits[:, -1, :]
                        
                        # Move to CPU for consistent processing
                        logits_cpu = logits.cpu()
                        probs = torch.nn.functional.softmax(logits_cpu, dim=-1)
                        model_probs.append(probs)
                        valid_indices.append(i)
                    except Exception as e:
                        print(f"Model {i} forward pass error: {e}")
                    
                if not valid_indices:
                    break
                    
            # Calculate union probabilities - all on CPU
            union_vocab_size = len(self.union_vocab_list)
            union_probs = torch.zeros(union_vocab_size, device="cpu")
            contrib_counts = torch.zeros(union_vocab_size, device="cpu")
            
            for i, model_idx in enumerate(valid_indices):
                probs = model_probs[i]
                
                # Move mapping to CPU if needed
                mapping = self.mappings[model_idx].cpu()
                
                # Get valid mappings and probabilities
                valid_mask = (mapping != -1) & (mapping < union_vocab_size)
                valid_model_indices = torch.nonzero(valid_mask, as_tuple=True)[0]
                valid_union_indices = mapping[valid_mask]
                
                if len(valid_model_indices) == 0:
                    continue
                    
                # Get probabilities for valid tokens
                model_token_probs = probs.squeeze(0)[valid_model_indices]
                
                # Safely add to union probabilities
                for j in range(len(valid_union_indices)):
                    if j < len(model_token_probs):  # Safety check
                        try:
                            union_idx = valid_union_indices[j].item()
                            prob = model_token_probs[j].item()
                            
                            if 0 <= union_idx < union_vocab_size:
                                union_probs[union_idx] += prob
                                contrib_counts[union_idx] += 1
                        except Exception as e:
                            print(f"Error adding probability: {e}")
            
            # Average probabilities
            valid_positions = contrib_counts > 0
            if torch.any(valid_positions):
                union_probs[valid_positions] /= contrib_counts[valid_positions]
            
            # Normalize
            prob_sum = union_probs.sum()
            if prob_sum > 0:
                union_probs /= prob_sum
            else:
                break
                
            # Apply temperature and top-p sampling
            if temperature == 0:
                # Greedy
                next_token_idx = torch.argmax(union_probs).item()
            else:
                # Apply temperature
                if temperature != 1.0:
                    logits = torch.log(torch.clamp(union_probs, min=1e-40))
                    logits = logits / temperature
                    union_probs = torch.softmax(logits, dim=-1)
                    
                # Top-p sampling
                sorted_probs, sorted_indices = torch.sort(union_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                mask = cumulative_probs > top_p
                mask[0] = False  # Keep at least one token
                
                # Zero out filtered tokens and renormalize
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum()
                
                # Sample
                sample_idx = torch.multinomial(sorted_probs, 1).item()
                next_token_idx = sorted_indices[sample_idx].item()
            
            # Get and decode token
            next_token = self.union_vocab_idx_to_token.get(next_token_idx)
            if next_token is None:
                break
                
            # Check if it's an EOS token
            if hasattr(self, 'eos_token_indices_in_union') and next_token_idx in self.eos_token_indices_in_union:
                break
                
            # Decode token
            decoded_token = ""
            for tokenizer in self.tokenizers:
                if next_token in tokenizer.get_vocab():
                    token_id = tokenizer.convert_tokens_to_ids(next_token)
                    decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
                    break
            
            if not decoded_token:
                decoded_token = next_token
                
            # Add to generated text
            generated_text += decoded_token
            generated_tokens_count += 1
            
            # Update input IDs
            new_input_ids_list = []
            for model_idx, (ids, tokenizer) in enumerate(zip(input_ids_list, self.tokenizers)):
                # Get the device for this model
                model_device = self.models[model_idx].device
                
                if next_token in tokenizer.get_vocab():
                    token_id = tokenizer.convert_tokens_to_ids(next_token)
                else:
                    token_id = tokenizer.unk_token_id
                    
                # Add token ID to input on the correct device
                token_tensor = torch.tensor([[token_id]], device=model_device)
                new_ids = torch.cat([ids, token_tensor], dim=-1)
                new_input_ids_list.append(new_ids)
                
            input_ids_list = new_input_ids_list
            
            # Check stop sequences
            for seq in stop_sequences:
                if generated_text.endswith(seq):
                    return generated_text

    def _get_single_model_predictions(self, context, model_idx, debug=False):
        """Get predictions from a single model"""
        try:
            model = self.models[model_idx]
            tokenizer = self.tokenizers[model_idx]
        except IndexError:
            if debug: 
                print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): IndexError accessing model/tokenizer. Models len: {len(self.models)}, Tokenizers len: {len(self.tokenizers)}")
            raise 
        except Exception as e_init:
            if debug: 
                print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Error during model/tokenizer init: {e_init}")
            raise 

        device = model.device
        context_str = str(context)

        if debug:
            context_preview = context_str[-70:]
            if len(context_str) > 70:
                context_preview = "..." + context_preview
            print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Context: '{context_preview}'")
            
            max_len_debug = getattr(model.config, "max_position_embeddings", getattr(tokenizer, 'model_max_length', 2048))
            print(f"        GMP_MAX_LEN (M{model_idx}): Effective max_length = {max_len_debug}")

        context_ids = tokenizer.encode(context_str, add_special_tokens=False, return_tensors="pt").to(device)
        
        max_len = getattr(model.config, 'max_position_embeddings', getattr(tokenizer, 'model_max_length', 2048))

        if context_ids.shape[1] >= max_len: 
            if debug: 
                print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Truncating context from {context_ids.shape[1]} to {max_len-1} tokens.")
            context_ids = context_ids[:, -(max_len-1):] 
        
        if context_ids.shape[1] == 0:
            if debug: 
                print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Context IDs empty after processing.")

        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs.logits[:, -1, :] 
            probs_tensor = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
        
        if debug: 
            print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Successfully got probs tensor shape {probs_tensor.shape}")
        return probs_tensor, tokenizer

    def _get_single_model_loglikelihood(self, context, continuation, model_idx, debug=False):
        """Get log-likelihood of continuation for a single model"""
        model = self.models[model_idx]
        tokenizer = self.tokenizers[model_idx]
        device = model.device

        # Tokenize exactly like lm-eval
        context_enc = tokenizer.encode(context, add_special_tokens=False)
        whole_enc = tokenizer.encode(context + continuation, add_special_tokens=False)
        
        # Continuation tokens = difference between whole and context
        cont_toks = whole_enc[len(context_enc):]
        
        if len(cont_toks) == 0:
            if debug:
                print(f"[Loglikelihood] No continuation tokens found")
            return -float('inf')

        # Input: whole sequence minus last token (standard causal LM setup)
        inp = torch.tensor([whole_enc[:-1]], device=device, dtype=torch.long)
        
        with torch.no_grad():
            outputs = model(inp)
            logits = outputs.logits  # [1, seq_len, vocab_size]
        
        # Select only continuation positions
        cont_len = len(cont_toks)
        logits = logits[:, -cont_len:, :tokenizer.vocab_size]
        
        # Get log probabilities and sum over continuation tokens
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        total_log_prob = 0.0
        for i, token_id in enumerate(cont_toks):
            if i < log_probs.shape[1] and token_id < log_probs.shape[2]:
                total_log_prob += log_probs[0, i, token_id].item()
        
        if debug:
            print(f"[Loglikelihood] Context len: {len(context_enc)}, Cont len: {cont_len}")
            print(f"[Loglikelihood] Continuation: '{continuation}'")
            print(f"[Loglikelihood] Cont tokens: {cont_toks}")
            print(f"[Loglikelihood] Log Prob: {total_log_prob:.4f}")

        return total_log_prob

    def _get_ensemble_token_probability(self, prefix_tokens, target_token, debug=False):
        """Get ensemble probability for a single token"""
        # Get weights from config with simple fallback
        weights = self.config.get("weights", [1.0/len(self.models)] * len(self.models))
        weights_np = np.array(weights) / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(self.models)) / len(self.models)
        
        # Collect probabilities from each model
        model_probs = []
        all_greedy_preds = []
        
        if debug:
            print(f"    Getting ensemble prob for token {target_token}")
        
        for model_idx in range(len(self.models)):
            try:
                model = self.models[model_idx]
                tokenizer = self.tokenizers[model_idx]
                device = model.device
                
                # Get single model probability
                log_prob = self._get_single_model_token_log_prob(
                    model, tokenizer, prefix_tokens, target_token, device
                )
                
                prob = math.exp(log_prob) if log_prob > -100 else 0.0
                model_probs.append((prob, weights_np[model_idx]))
                
                # Get greedy prediction for ensemble voting
                if len(prefix_tokens) > 0:
                    with torch.no_grad():
                        input_ids = torch.tensor([prefix_tokens], device=device)
                        max_len = getattr(model.config, 'max_position_embeddings', 2048)
                        if input_ids.shape[1] >= max_len:
                            input_ids = input_ids[:, -(max_len-1):]
                        outputs = model(input_ids)
                        greedy_pred = torch.argmax(outputs.logits[0, -1, :]).item()
                        all_greedy_preds.append(greedy_pred)
                
                if debug and model_idx == 0:
                    print(f"      Model {model_idx}: log_prob={log_prob:.4f}, prob={prob:.6f}")
                    
            except Exception as e:
                if debug:
                    print(f"      Model {model_idx} error: {e}")
                model_probs.append((0.0, weights_np[model_idx]))
        
        # Calculate weighted ensemble probability
        if not model_probs:
            return -float('inf'), False
        
        ensemble_prob = sum(prob * weight for prob, weight in model_probs)
        
        # Convert back to log space
        if ensemble_prob > 0:
            ensemble_log_prob = math.log(ensemble_prob)
        else:
            ensemble_log_prob = -float('inf')
        
        # Check if target is greedy (majority vote)
        is_greedy = False
        if all_greedy_preds:
            greedy_vote = {}
            for pred in all_greedy_preds:
                greedy_vote[pred] = greedy_vote.get(pred, 0) + 1
            most_common = max(greedy_vote, key=greedy_vote.get)
            is_greedy = (most_common == target_token)
        
        if debug:
            print(f"      Ensemble: prob={ensemble_prob:.6f}, log_prob={ensemble_log_prob:.4f}")
        
        return ensemble_log_prob, is_greedy

    def _get_single_model_token_log_prob(self, model, tokenizer, prefix_tokens, target_token, device):
        """Get log probability from a single model for a single token"""
        # Handle empty prefix
        if not prefix_tokens:
            if tokenizer.bos_token_id is not None:
                prefix_tokens = [tokenizer.bos_token_id]
            else:
                return -50.0  # Arbitrary penalty for no context
        
        # Create input tensor
        input_ids = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
        
        # Truncate if needed
        max_len = getattr(model.config, 'max_position_embeddings', 2048)
        if input_ids.shape[1] >= max_len:
            input_ids = input_ids[:, -(max_len-1):]
        
        # Get model output
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last position
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Get target token log probability
        if target_token < log_probs.shape[-1]:
            return log_probs[0, target_token].item()
        else:
            return -float('inf')