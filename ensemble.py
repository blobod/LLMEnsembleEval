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
                config.weights
                if config.weights
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
        for idx, name in enumerate(self.config.model_names):
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
                    
                    # Check GPU memory before loading
                    mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    mem_total = torch.cuda.get_device_properties(device).total_memory / 1e9
                    print(f"    GPU {device} memory before loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved, {mem_total:.2f}GB total")
                    
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
                    use_auth_token=self.config.token,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,   
                    low_cpu_mem_usage=True
                ).to(device)  # Use .to() after loading
                
                self.models.append(model)
                
                # Check GPU memory after loading
                if device.type == 'cuda':
                    mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                    mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                    print(f"    GPU {device} memory after loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                
                print(f"    Model {name} loaded on {device}.")
            except Exception as e:
                print(f"    ERROR loading model {name}: {e}")
                import traceback
                print(f"    Full traceback: {traceback.format_exc()}")
                
                # Try alternate loading method
                try:
                    print(f"    Attempting to load with CPU first, then transfer...")
                    # Load to CPU first
                    model = AutoModelForCausalLM.from_pretrained(
                        name,
                        use_auth_token=self.config.token,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,   
                        low_cpu_mem_usage=True,
                        device_map="cpu",
                    )
                    # Then transfer to target device
                    model = model.to(device)
                    self.models.append(model)
                    
                    # Check GPU memory after loading with alternate method
                    if device.type == 'cuda':
                        mem_allocated = torch.cuda.memory_allocated(device) / 1e9
                        mem_reserved = torch.cuda.memory_reserved(device) / 1e9
                        print(f"    GPU {device} memory after CPU->GPU loading: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
                        
                    print(f"    Model {name} loaded on {device} (CPU->GPU transfer).")
                except Exception as e2:
                    print(f"    Second attempt failed: {e2}")
                    raise e  # Raise the original error

    def _load_tokenizers(self):
        """Load tokenizers for all models."""
        print("[GAC __init__] Initializing GAC Tokenizers...")
        for name in self.config.tokenizer_names:
            print(f"  Loading tokenizer: {name}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    name, use_auth_token=self.config.token, trust_remote_code=True
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
        """Create a UNION vocabulary with special handling for different tokenizer types."""
        print("[GAC _create_vocab] Creating improved UNION vocabulary...")
        
        # Initialize for the union vocabulary
        self.union_vocab_set = set()
        
        # Process each tokenizer to find dominant prefix type
        g_prefix_count = 0
        u_prefix_count = 0
        
        for tokenizer in self.tokenizers:
            vocab = tokenizer.get_vocab()
            g_prefix_count += sum(token.startswith("Ġ") for token in vocab)
            u_prefix_count += sum(token.startswith("▁") for token in vocab)
        
        # Determine dominant prefix type
        uses_g_prefix = g_prefix_count >= u_prefix_count
        print(f"  Detected prefix types: Ġ={g_prefix_count}, ▁={u_prefix_count}")
        print(f"  Using dominant prefix type: {'Ġ' if uses_g_prefix else '▁'}")
        
        # Process each tokenizer with normalized tokens
        for i, tokenizer in enumerate(self.tokenizers):
            try:
                vocab = tokenizer.get_vocab()
                normalized_tokens = set()
                
                for token in vocab:
                    # Normalize tokens based on dominant prefix
                    if uses_g_prefix:
                        # Replace ▁ with Ġ for consistency if this token uses ▁
                        if token.startswith("▁"):
                            normalized = "Ġ" + token[1:]
                        else:
                            normalized = token
                    else:
                        # Replace Ġ with ▁ for consistency if this token uses Ġ
                        if token.startswith("Ġ"):
                            normalized = "▁" + token[1:]
                        else:
                            normalized = token
                    
                    normalized_tokens.add(normalized)
                
                # Add normalized tokens to union vocabulary
                self.union_vocab_set.update(normalized_tokens)
                print(f"  Added {len(normalized_tokens)} normalized tokens from tokenizer {i}")
                
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
        
        # Store the dominant prefix type for later use
        self.uses_g_prefix = uses_g_prefix
        
        print(f"  Union vocabulary size: {len(self.union_vocab_list)}")
        if not self.union_vocab_list:
            print("[ERROR] Union vocabulary is empty!")

    def _create_mappings(self):
        """Create mappings from each model's vocabulary to the UNION vocabulary indices."""
        print("[GAC _create_mappings] Creating improved mappings to UNION vocabulary...")
        self.mappings = []
        
        if self.union_vocab_token_to_idx is None:
            raise RuntimeError("Union vocabulary must be created before mappings.")
        
        for i, (model, tokenizer) in enumerate(zip(self.models, self.tokenizers)):
            try:
                model_vocab_size = model.config.vocab_size
                print(f"  Mapping for model {i} (Vocab size: {model_vocab_size}) → Union Size {len(self.union_vocab_list)}")
                
                # CRITICAL FIX: Create mapping array on CPU first
                mapping = torch.full((model_vocab_size,), -1, dtype=torch.long, device="cpu")
                
                # Get tokenizer vocabulary
                vocab = tokenizer.get_vocab()
                mapped_count = 0
                
                # Process each token in this tokenizer's vocabulary
                for token, token_id in vocab.items():
                    if token_id >= model_vocab_size:
                        continue  # Skip tokens with IDs outside the model's vocab size
                    
                    # Try direct mapping first
                    if token in self.union_vocab_token_to_idx:
                        union_idx = self.union_vocab_token_to_idx[token]
                        mapping[token_id] = union_idx
                        mapped_count += 1
                    else:
                        # Try with normalized prefix
                        normalized = None
                        
                        if hasattr(self, 'uses_g_prefix'):
                            if self.uses_g_prefix:
                                # Normalize to Ġ
                                if token.startswith("▁"):
                                    normalized = "Ġ" + token[1:]
                                elif not token.startswith("Ġ") and token.strip() and not token[0].isalnum():
                                    # Add Ġ for tokens that should have it but don't
                                    normalized = "Ġ" + token
                            else:
                                # Normalize to ▁
                                if token.startswith("Ġ"):
                                    normalized = "▁" + token[1:]
                                elif not token.startswith("▁") and token.strip() and not token[0].isalnum():
                                    # Add ▁ for tokens that should have it but don't
                                    normalized = "▁" + token
                        
                        # Try the normalized version
                        if normalized and normalized in self.union_vocab_token_to_idx:
                            union_idx = self.union_vocab_token_to_idx[normalized]
                            mapping[token_id] = union_idx
                            mapped_count += 1
                
                print(f"    Mapped {mapped_count}/{len(vocab)} tokens")
                
                # IMPORTANT: Move the mapping to the device where the model is
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
        
        return generated_text

    @staticmethod
    def get_gpu_info():
        """Get information about available GPUs."""
        import os
        import torch
        
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")