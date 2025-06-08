import torch
import math
import os
import sys
import json
import traceback
from typing import List, Optional, Dict
import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from ensemble import GenerationAsClassification


@register_model("gac_ensemble_wrapper")
class EnsembleHarnessWrapper(LM):
    """
    Wrapper class that serves as an adapter between the Ensemble and lm-evaluation-harness,
    implementing the evaluation framework's expected interface methods.
    """
    
    def __init__(
        self,
        config_path: str = None,
        batch_size: int = 1,
        device: Optional[str] = None,
        devices: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self._batch_size = 1

        # Handle config_path parameter (now optional)
        if config_path is None:
            # Try to get config_path from environment variable or command line args
            config_path = os.environ.get("GAC_CONFIG_PATH", None)
            if config_path is None:
                # Try to find config in the current directory or /app/work
                possible_configs = [
                    "gac_config.json",
                    "gac_config_one_model.json",
                    "/app/work/gac_config.json",
                    "/app/work/gac_config_one_model.json",
                ]
                for cfg in possible_configs:
                    if os.path.exists(cfg):
                        config_path = cfg
                        print(f"[INFO] Found config at: {config_path}")
                        break

            if config_path is None:
                raise ValueError(
                    "No config_path provided and no config found. Please provide a config path."
                )

        # Check if the config exists at different paths
        try:
            if os.path.exists(config_path):
                print(f"[INFO] Config found at original path: {config_path}")
            elif os.path.exists(os.path.basename(config_path)):
                print(
                    f"[INFO] Config found with basename: {os.path.basename(config_path)}"
                )
                config_path = os.path.basename(config_path)
            elif os.path.exists(
                os.path.join(os.getcwd(), os.path.basename(config_path))
            ):
                print(
                    f"[INFO] Config found in current working directory: {os.path.join(os.getcwd(), os.path.basename(config_path))}"
                )
                config_path = os.path.join(os.getcwd(), os.path.basename(config_path))
            elif os.path.exists("/app/work/" + os.path.basename(config_path)):
                print(
                    f"[INFO] Config found at /app/work/: {'/app/work/' + os.path.basename(config_path)}"
                )
                config_path = "/app/work/" + os.path.basename(config_path)
            else:
                raise FileNotFoundError(
                    f"Config not found at any of the checked paths: {config_path}"
                )

            print(f"[INFO] Using config from: {config_path}")
        except Exception as e:
            print(f"[ERROR] Exception during config check: {e}")
            raise

        # Load Config from JSON
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        print(f"Loaded config: {config_dict}")
        print(f"Devices in config: {config_dict.get('devices', 'Not found')}")
        
        # Set up device mapping
        if devices is not None:
            self._devices = [torch.device(d) for d in devices]
            print(f"Using multiple devices: {self._devices}")
        else:
            # Original single device logic
            if device is not None:
                self._device = torch.device(device)
                config_dict["device"] = device
            elif "device" in config_dict:
                self._device = torch.device(config_dict["device"])
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                config_dict["device"] = str(self._device)
            self._devices = [self._device]  # Make a list for consistent handling
            print(f"Using single device: {self._device}")
        
        self._device = self._devices[0]
        print(f"DEBUG - Initialized devices: {self._devices}")
        
        # Modify config to use multiple devices
        config_dict.setdefault("trust_remote_code", True)
        config_dict["model_count"] = len(config_dict.get("model_names", []))
        
        # Set devices in config
        config_dict["devices"] = [str(d) for d in self._devices]
        
        # Initialize GAC with the config dict directly
        self.gac = GenerationAsClassification(config_dict)
        
        # Prepare the ensemble
        print(f"[DEBUG] GAC models: {hasattr(self.gac, 'models')} (length: {len(self.gac.models) if hasattr(self.gac, 'models') else 'N/A'})")
        print(f"[DEBUG] GAC tokenizers: {hasattr(self.gac, 'tokenizers')} (length: {len(self.gac.tokenizers) if hasattr(self.gac, 'tokenizers') else 'N/A'})")

        # Check models and tokenizers before prepare
        if (hasattr(self.gac, 'models') and self.gac.models and 
            hasattr(self.gac, 'tokenizers') and self.gac.tokenizers):
            # Safe to prepare
            self.gac.prepare(self.gac.models, self.gac.tokenizers, self._devices)
        else:
            print("[WARN] Models or tokenizers missing, skipping prepare step")
            if not hasattr(self.gac, 'models') or not self.gac.models:
                print("[ERROR] No models loaded!")
            if not hasattr(self.gac, 'tokenizers') or not self.gac.tokenizers:
                print("[ERROR] No tokenizers loaded!")
        
        # Use first model's config for max_length approximation
        self.tokenizer = self.gac.tokenizers[0]
        self._model_config = self.gac.models[0].config
        print("[Wrapper] Ensemble Wrapper Initialized (Multi-GPU).")

    # --- Properties ---
    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def max_batch_size(self):
        return self._batch_size

    @property
    def max_length(self):
        ml = None
        if hasattr(self._model_config, "max_position_embeddings"):
            ml = self._model_config.max_position_embeddings
        elif hasattr(self._model_config, "n_positions"):
            ml = self._model_config.n_positions
        elif hasattr(self._model_config, "max_sequence_length"):
            ml = self._model_config.max_sequence_length
        if ml is None:
            print("[WARN] max_length could not be determined. Defaulting to 2048.")
            ml = 2048
        return ml
    
    @property
    def eot_token_id(self):
        """Return the EOT token ID for the model."""
        if hasattr(self.gac, 'primary_eos_token') and self.gac.primary_eos_token:
            # Use the primary EOS token from GAC
            return self.tokenizer.convert_tokens_to_ids(self.gac.primary_eos_token)
        else:
            # Fallback to tokenizer's EOS token
            return self.tokenizer.eos_token_id

    def _get_ensemble_weights(self):
        num_models = len(self.gac.models)
        if hasattr(self.gac, 'weights') and self.gac.weights is not None and \
        len(self.gac.weights) == num_models:

            weights_tensor = self.gac.weights
            # Ensure weights_tensor is at least 1D before converting to numpy
            if weights_tensor.ndim == 0: # If it's a scalar tensor
                weights_tensor = weights_tensor.unsqueeze(0) # Make it 1D, e.g., tensor([1.0])

            weights_np = weights_tensor.cpu().numpy().squeeze() # squeeze is okay if it's already 1D or higher

            # Ensure weights_np is at least 1D after squeeze
            if weights_np.ndim == 0:
                weights_np = np.array([weights_np.item()]) # Convert 0D array to 1D array, e.g., array(1.0) -> array([1.0])

            # Normalize
            if np.sum(weights_np) == 0:
                weights_np = np.ones(num_models) / num_models
            else:
                weights_np = weights_np / np.sum(weights_np)
        else:
            weights_np = np.ones(num_models) / num_models

        # Final check to ensure it's 1D
        if weights_np.ndim == 0 and num_models == 1:
            return np.array([weights_np.item()])
        elif weights_np.ndim == 0 and num_models > 1: # Should not happen with above logic
            return np.ones(num_models) / num_models

        return weights_np # Should now always be a 1D numpy array

    def _get_model_predictions(self, context, model_idx, debug=False):
        # Ensure model and tokenizer are assigned first
        try:
            model = self.gac.models[model_idx]
            tokenizer = self.gac.tokenizers[model_idx]
        except IndexError:
            if debug: 
                print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): IndexError accessing model/tokenizer. Models len: {len(self.gac.models)}, Tokenizers len: {len(self.gac.tokenizers)}")
                sys.stdout.flush()
            raise 
        except Exception as e_init:
            if debug: print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Error during model/tokenizer init: {e_init}"); sys.stdout.flush()
            raise 

        device = model.device
        context_str = str(context)

        if debug:
            context_preview = context_str[-70:] # Take the last 70 characters
            if len(context_str) > 70:
                context_preview = "..." + context_preview # Prepend ... if original was longer
            print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Context: '{context_preview}'")
            
            # Optional: Print max_len being used
            max_len_debug = getattr(model.config, "max_position_embeddings", getattr(tokenizer, 'model_max_length', 2048))
            print(f"        GMP_MAX_LEN (M{model_idx}): Effective max_length = {max_len_debug}")
            sys.stdout.flush()

        context_ids = tokenizer.encode(context_str, add_special_tokens=False, return_tensors="pt").to(device)
        
        max_len = getattr(model.config, 'max_position_embeddings', getattr(tokenizer, 'model_max_length', 2048))

        if context_ids.shape[1] >= max_len: 
            if debug: print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Truncating context from {context_ids.shape[1]} to {max_len-1} tokens.")
            context_ids = context_ids[:, -(max_len-1):] 
        
        if context_ids.shape[1] == 0:
            if debug: print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Context IDs empty after processing. Model might error or produce poor logits.")

        with torch.no_grad():
            outputs = model(context_ids)
            logits = outputs.logits[:, -1, :] 
            probs_tensor = torch.nn.functional.softmax(logits, dim=-1).squeeze(0) # Squeeze batch dim
        
        if debug: print(f"      GET_MODEL_PREDICTIONS (M{model_idx}): Successfully got probs tensor shape {probs_tensor.shape}")
        return probs_tensor, tokenizer

    def _get_loglikelihood_of_continuation(self, context, continuation, model_idx, debug=False):
        """
        Computes log-likelihood of continuation given context using a single model.
        MATCHES lm-evaluation-harness implementation exactly.
        """
        model = self.gac.models[model_idx]
        tokenizer = self.gac.tokenizers[model_idx]
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
        
        # CRITICAL: Select only continuation positions
        # If context is tokens [0,1,2,3] and continuation is [4,5,6]
        # Then logits positions 3,4,5 predict tokens 4,5,6
        cont_len = len(cont_toks)
        inplen = len(context_enc)
        
        # This matches lm-eval's _select_cont_toks for causal models
        logits = logits[:, -cont_len:, :tokenizer.vocab_size]  # Select continuation positions
        
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

    ## Benchmark Specific Handlers
    def _handle_mmlu(self, context, continuation, instance_doc, debug=False):
        expected_answer_char = str(continuation).strip().upper()
        if expected_answer_char not in ["A", "B", "C", "D"]:
            if debug: print(f"MMLU WARN: Unexpected answer format '{continuation}'")
            return -float('inf'), False

        weights = self._get_ensemble_weights()
        ensemble_choice_probs = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        
        for model_idx in range(len(self.gac.models)):
            try:
                probs_after_context, tokenizer = self._get_model_predictions(context, model_idx)
                model_choice_probs = {}
                for choice_char_option in ["A", "B", "C", "D"]:
                    # Check P(choice_char_option) and P(" " + choice_char_option)
                    prob_no_space, prob_with_space = 0.0, 0.0
                    
                    tok_id_no_space = tokenizer.encode(choice_char_option, add_special_tokens=False)
                    if tok_id_no_space and len(tok_id_no_space) == 1 and tok_id_no_space[0] < probs_after_context.shape[0]:
                        prob_no_space = probs_after_context[tok_id_no_space[0]].item()
                    
                    tok_id_with_space = tokenizer.encode(" " + choice_char_option, add_special_tokens=False)
                    if tok_id_with_space and len(tok_id_with_space) == 1 and tok_id_with_space[0] < probs_after_context.shape[0]:
                        prob_with_space = probs_after_context[tok_id_with_space[0]].item()
                    
                    model_choice_probs[choice_char_option] = max(prob_no_space, prob_with_space)
                    if debug: print(f"    M{model_idx} MMLU '{choice_char_option}': P(no_sp)={prob_no_space:.3f}, P(w_sp)={prob_with_space:.3f} -> ChosenP={model_choice_probs[choice_char_option]:.3f}")
                
                for choice_c, prob_c in model_choice_probs.items():
                    ensemble_choice_probs[choice_c] += prob_c * weights[model_idx]
            except Exception as e:
                if debug: print(f"  MMLU M{model_idx} ERR: {e}")

        final_log_prob_of_continuation = -float('inf')
        continuation_prob = ensemble_choice_probs.get(expected_answer_char, 0.0)
        if continuation_prob > 1e-9:
            final_log_prob_of_continuation = math.log(continuation_prob)
        
        predicted_choice_char = ""
        if ensemble_choice_probs: # Check if not empty
             predicted_choice_char = max(ensemble_choice_probs, key=ensemble_choice_probs.get, default="")

        is_greedy = (predicted_choice_char == expected_answer_char)

        if debug:
            print(f"  MMLU Ensemble Probs: { {k: f'{v:.3f}' for k,v in ensemble_choice_probs.items()} }")
            print(f"  MMLU Expected: '{expected_answer_char}', Pred: '{predicted_choice_char}', LogProbCont: {final_log_prob_of_continuation:.3f}, IsGreedy: {is_greedy}")
        return final_log_prob_of_continuation, is_greedy

    def _handle_piqa(self, context, continuation, instance_doc, debug=False):
        if debug: 
            print(f"    PIQA HANDLER: Started. Cont: '{str(continuation)[:30]}...'")
            sys.stdout.flush()

        weights = self._get_ensemble_weights()
        sum_weighted_log_probs = 0.0

        model_details_for_debug = [] 

        for model_idx in range(len(self.gac.models)):
            model_name = f"M{model_idx}"
            model_log_prob = -float('inf')
            try:
                model_log_prob = self._get_loglikelihood_of_continuation(
                    context, continuation, model_idx,
                    debug=debug
                )
                
                if model_log_prob > -float('inf'): 
                    sum_weighted_log_probs += model_log_prob * weights[model_idx]
                
                if debug:
                    model_prob = math.exp(model_log_prob) if model_log_prob > -float('inf') else 0.0
                    model_details_for_debug.append(
                        f"M{model_idx}: LogProb={model_log_prob:.4f}, Prob={model_prob:.4f}, Weight={weights[model_idx]:.4f}"
                    )

            except Exception as e:
                if debug: 
                    print(f"    PIQA HANDLER {model_name} ERR: {e}")
                    model_details_for_debug.append(f"M{model_idx}: ERROR ({e})")
        
        final_ensemble_log_prob = sum_weighted_log_probs

        if debug:
            print(f"    PIQA HANDLER: Model Contributions for cont '{str(continuation)[:30]}...':")
            for detail in model_details_for_debug:
                print(f"      {detail}")
            final_ensemble_prob = math.exp(final_ensemble_log_prob) if final_ensemble_log_prob > -float('inf') else 0.0
            print(f"    PIQA HANDLER: Ensemble Final LogProb={final_ensemble_log_prob:.4f}, Final Prob={final_ensemble_prob:.4f}")
            print(f"    PIQA HANDLER: Finished.")
            sys.stdout.flush()
            
        return float(final_ensemble_log_prob), False

    def _handle_arc(self, context, continuation, instance_doc, debug=False):
        """ARC handler with comprehensive debugging"""
        if debug:
            print("\n" + "="*80)
            print("🧪 ARC HANDLER DEBUG")
            print("="*80)
            
            # Show the full question context
            print(f"📝 CONTEXT (last 200 chars):")
            context_preview = str(context)[-200:] if len(str(context)) > 200 else str(context)
            print(f"   '{context_preview}'")
            
            # Show what we're evaluating
            print(f"\n🎯 CONTINUATION BEING EVALUATED:")
            print(f"   '{continuation}'")
            
            # Show the correct answer
            gold_label = instance_doc.get("answerKey", "").strip().upper()
            print(f"\n✅ CORRECT ANSWER: '{gold_label}'")
            
            # Show all choices if available
            if 'choices' in instance_doc:
                print(f"\n📋 ALL CHOICES:")
                choices = instance_doc['choices']
                if isinstance(choices, dict) and 'label' in choices:
                    for i, (label, text) in enumerate(zip(choices['label'], choices['text'])):
                        marker = "👉" if label.upper() == gold_label else "  "
                        print(f"   {marker} {label}: {text}")
                elif isinstance(choices, list):
                    for choice in choices:
                        if isinstance(choice, dict):
                            label = choice.get('label', '?')
                            text = choice.get('text', '?')
                            marker = "👉" if label.upper() == gold_label else "  "
                            print(f"   {marker} {label}: {text}")
            
            print(f"\n🤖 MODEL PREDICTIONS:")

        weights = self._get_ensemble_weights()
        sum_weighted_log_probs = 0.0
        model_results = []

        for model_idx in range(len(self.gac.models)):
            try:
                model_log_prob = self._get_loglikelihood_of_continuation(
                    context, continuation, model_idx, debug=False  # Reduce noise
                )

                if model_log_prob > -float('inf'):
                    sum_weighted_log_probs += model_log_prob * weights[model_idx]
                    
                # Store results for debugging
                model_results.append({
                    'model_idx': model_idx,
                    'log_prob': model_log_prob,
                    'prob': math.exp(model_log_prob) if model_log_prob > -100 else 0.0,
                    'weight': weights[model_idx],
                    'weighted_contribution': model_log_prob * weights[model_idx] if model_log_prob > -float('inf') else 0.0
                })

                if debug:
                    prob = math.exp(model_log_prob) if model_log_prob > -100 else 0.0
                    print(f"   M{model_idx}: log_prob={model_log_prob:.4f}, prob={prob:.6f}, weight={weights[model_idx]:.3f}")

            except Exception as e:
                model_results.append({
                    'model_idx': model_idx,
                    'log_prob': -float('inf'),
                    'prob': 0.0,
                    'weight': weights[model_idx],
                    'error': str(e)
                })
                if debug:
                    print(f"   M{model_idx}: ERROR - {e}")

        # Calculate final prediction by finding which choice matches the continuation
        predicted_label = "UNKNOWN"
        gold_label = instance_doc.get("answerKey", "").strip().upper()
        
        # Find which choice this continuation corresponds to
        if 'choices' in instance_doc:
            choices = instance_doc['choices']
            continuation_clean = str(continuation).strip().lower()
            
            if isinstance(choices, dict) and 'label' in choices and 'text' in choices:
                for label, text in zip(choices['label'], choices['text']):
                    if text.strip().lower() in continuation_clean or continuation_clean in text.strip().lower():
                        predicted_label = label.strip().upper()
                        break
            elif isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        label = choice.get('label', '').strip().upper()
                        text = choice.get('text', '').strip().lower()
                        if text in continuation_clean or continuation_clean in text:
                            predicted_label = label
                            break
        
        is_correct = predicted_label == gold_label
        
        ensemble_prob = math.exp(sum_weighted_log_probs) if sum_weighted_log_probs > -100 else 0.0

        if debug:
            print(f"\n📊 ENSEMBLE RESULTS:")
            print(f"   Ensemble log_prob: {sum_weighted_log_probs:.4f}")
            print(f"   Ensemble prob: {ensemble_prob:.6f}")
            print(f"   Predicted: '{predicted_label}' vs Gold: '{gold_label}'")
            print(f"   ✅ Correct: {is_correct}")
            
            # Show model contributions
            print(f"\n🔍 MODEL CONTRIBUTIONS:")
            for result in model_results:
                if 'error' in result:
                    print(f"   M{result['model_idx']}: ERROR - {result['error']}")
                else:
                    contribution = result['weighted_contribution']
                    print(f"   M{result['model_idx']}: {contribution:.4f} (log_prob={result['log_prob']:.4f} × weight={result['weight']:.3f})")
            
            print("="*80 + "\n")

        return float(sum_weighted_log_probs), is_correct

    def _handle_winogrande(self, context, continuation, instance_doc, debug=False):
        """Handle WinoGrande as sentence completion (current lm-eval format)"""
        if debug:
            print(f"\n🔤 WINOGRANDE (Sentence Completion Mode)")
            print(f"📝 Full sentence: {instance_doc.get('sentence', 'N/A')}")
            print(f"🎯 Context: '{context}'")
            print(f"🎯 Continuation: '{continuation}'")
            
            # Determine which option this represents
            option1 = instance_doc.get('option1', '')
            option2 = instance_doc.get('option2', '')
            correct = instance_doc.get('answer', '')
            
            # Check which option is in the context
            current_option = "UNKNOWN"
            if option1 in context:
                current_option = "1"
            elif option2 in context:
                current_option = "2"
                
            print(f"📋 Option1: '{option1}', Option2: '{option2}'")
            print(f"✅ Correct: Option {correct}")
            print(f"🎯 This context has: Option {current_option}")

        # Calculate P(continuation | context) as before
        weights = self._get_ensemble_weights()
        sum_weighted_log_probs = 0.0

        for model_idx in range(len(self.gac.models)):
            try:
                model_log_prob = self._get_loglikelihood_of_continuation(
                    context, continuation, model_idx, debug=False
                )
                
                if model_log_prob > -float('inf'): 
                    sum_weighted_log_probs += model_log_prob * weights[model_idx]
                    
                if debug:
                    print(f"   M{model_idx}: log_prob={model_log_prob:.4f}")
                    
            except Exception as e:
                if debug: 
                    print(f"   M{model_idx}: ERROR - {e}")

        # Determine correctness
        option1 = instance_doc.get('option1', '')
        option2 = instance_doc.get('option2', '')
        correct_option = instance_doc.get('answer', '')
        
        current_option = "UNKNOWN"
        if option1 in context:
            current_option = "1"
        elif option2 in context:
            current_option = "2"
        
        is_correct = current_option == correct_option

        if debug:
            print(f"📊 Result: log_prob={sum_weighted_log_probs:.4f}, correct={is_correct}")

        return float(sum_weighted_log_probs), is_correct

    ## Benchmark Detection
    def _detect_benchmark_from_instance(self, instance, debug=False):
        """Detects benchmark type from instance attributes with strict error handling."""
        task_name_on_instance = None
        if hasattr(instance, '_task_name') and getattr(instance, '_task_name'):
            task_name_on_instance = getattr(instance, '_task_name')
        elif hasattr(instance, 'task_name') and getattr(instance, 'task_name'):
            task_name_on_instance = getattr(instance, 'task_name')
        
        print(f"TASK DEBUG: task_name='{task_name_on_instance}'")
        print(f"TASK DEBUG: instance attributes: {[attr for attr in dir(instance) if 'task' in attr.lower()]}")
        
        if task_name_on_instance:
            task_name_lower = task_name_on_instance.lower()
            print(f"TASK DEBUG: task_name_lower='{task_name_lower}'")
            
            if "mmlu" in task_name_lower: 
                print("TASK DEBUG: Detected MMLU -> routing to MMLU")
                return "mmlu"
            if "piqa" in task_name_lower: 
                return "piqa"
            if "arc" in task_name_lower:
                print("TASK DEBUG: Detected Arc -> routing to Arc")
                return "arc"
            if "winogrande" in task_name_lower:
                print("TASK DEBUG: Detected WinoGrande -> routing to WinoGrande")
                return "winogrande"
        
        # Check doc structure for additional clues
        doc = getattr(instance, 'doc', {})
        context, continuation = instance.args
        
        print(f"TASK DEBUG: doc keys: {list(doc.keys()) if doc else 'None'}")
        print(f"TASK DEBUG: continuation: '{str(continuation)[:50]}...'")
        print(f"TASK DEBUG: context preview: '{str(context)[-100:]}'")
        
        # Check continuation format as last resort
        continuation_clean = str(continuation).strip().upper()
        if continuation_clean in ["A", "B", "C", "D"] and len(continuation_clean) == 1:
            print("TASK DEBUG: Detected A/B/C/D continuation -> routing to MMLU")
            return "mmlu"
        
        # STRICT: If we can't detect the benchmark, raise an error
        raise ValueError(f"UNKNOWN BENCHMARK TYPE! "
                        f"task_name='{task_name_on_instance}', "
                        f"doc_keys={list(doc.keys()) if doc else 'None'}, "
                        f"continuation='{continuation}', "
                        f"instance_type={type(instance)}")

    def loglikelihood(self, requests):
        """Calculate log-likelihood for a batch of requests."""
        if not requests:
            return []

        results = []
        
        for idx, instance in enumerate(requests):
            try:
                # Validate instance structure
                if not (hasattr(instance, 'args') and isinstance(instance.args, tuple) and len(instance.args) == 2):
                    results.append((-float('inf'), False))
                    continue 

                context, continuation = instance.args
                doc = getattr(instance, 'doc', {})
                
                # Detect benchmark type and call appropriate handler
                detected_type = self._detect_benchmark_from_instance(instance, debug=False)
                
                if detected_type == "piqa":
                    log_prob, is_greedy = self._handle_piqa(context, continuation, doc, debug=True)
                elif detected_type == "mmlu":
                    log_prob, is_greedy = self._handle_mmlu(context, continuation, doc, debug=True)
                elif detected_type == "arc":
                    log_prob, is_greedy = self._handle_arc(context, continuation, doc, debug=True)
                elif detected_type == "winogrande":
                    log_prob, is_greedy = self._handle_winogrande(context, continuation, doc, debug=True)
                else:
                    # Raise error for unknown benchmark types to ensure proper handling
                    raise ValueError(f"Unknown benchmark type detected: {detected_type}. "
                                    f"Supported benchmarks: piqa, mmlu, arc, winogrande")

                results.append((float(log_prob), is_greedy))
                
            except Exception as e:
                # Log error for debugging but continue processing
                print(f"Error processing instance {idx}: {e}")
                results.append((-float('inf'), False))

        return results
    
    def loglikelihood_rolling(self, requests):
        """
        Compute the log-likelihood of each token in the context, conditioned on the previous tokens.
        Simplified version that closely matches what lm-eval expects.
        """
        print("[Wrapper] Calculate rolling loglikelihoods...")
        results = []
        
        for request_idx, request in enumerate(requests):
            context = request.args[0]
            token_loglikelihoods = []
            token_is_greedy = []
            
            try:
                # Get tokens - handle different input types
                if isinstance(context, str):
                    # For string input, encode it
                    all_tokens = self.tokenizer.encode(context, add_special_tokens=False)
                elif isinstance(context, list):
                    # Already tokenized
                    all_tokens = context
                elif isinstance(context, torch.Tensor):
                    # Convert tensor to list
                    all_tokens = context.tolist()
                else:
                    print(f"[WARN] Unknown context type: {type(context)}")
                    all_tokens = []
                
                # Debug info for first few
                if request_idx < 3:
                    print(f"\n[DEBUG] Request {request_idx}:")
                    print(f"  Total tokens: {len(all_tokens)}")
                    if len(all_tokens) > 0:
                        print(f"  First 10 tokens: {all_tokens[:10]}")
                        print(f"  Text preview: '{self.tokenizer.decode(all_tokens[:50])}...'")
                
                # Calculate log likelihood for each token given previous tokens
                for i in range(len(all_tokens)):
                    if i == 0:
                        # First token - use BOS if available, otherwise skip
                        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                            prefix = [self.tokenizer.bos_token_id]
                            target = all_tokens[0]
                            log_prob, is_greedy = self._get_token_log_prob_ensemble(prefix, target, debug=(request_idx < 3 and i < 5))
                            token_loglikelihoods.append(log_prob)
                            token_is_greedy.append(is_greedy)
                        # If no BOS, skip first token (no context to condition on)
                        continue
                    
                    # For other tokens: condition on all previous tokens
                    prefix = all_tokens[:i]
                    target = all_tokens[i]
                    
                    # Get ensemble log probability
                    log_prob, is_greedy = self._get_token_log_prob_ensemble(
                        prefix, target, debug=(request_idx < 3 and i < 5)
                    )
                    
                    token_loglikelihoods.append(log_prob)
                    token_is_greedy.append(is_greedy)
                
                # Debug summary
                if request_idx < 3 and token_loglikelihoods:
                    avg_log_prob = sum(token_loglikelihoods) / len(token_loglikelihoods)
                    print(f"  Results: {len(token_loglikelihoods)} log probs")
                    print(f"  Average log prob: {avg_log_prob:.4f}")
                    print(f"  Perplexity estimate: {math.exp(-avg_log_prob):.2f}")
                
                results.append((token_loglikelihoods, token_is_greedy))
                
            except Exception as e:
                print(f"[ERROR] Failed to process request {request_idx}: {e}")
                import traceback
                traceback.print_exc()
                results.append(([], []))
        
        print(f"[Wrapper] Rolling loglikelihoods complete. Processed {len(results)} requests.")
        return results

    def _get_token_log_prob_ensemble(self, prefix_tokens, target_token, debug=False):
        """
        Get ensemble log probability for a single token.
        This method properly handles the ensemble calculation.
        """
        weights = self._get_ensemble_weights()
        
        # Collect probabilities (not log probabilities) from each model
        model_probs = []
        all_greedy_preds = []
        
        if debug:
            print(f"    Getting ensemble prob for token {target_token} ('{self.tokenizer.decode([target_token])}')")
        
        for model_idx in range(len(self.gac.models)):
            try:
                model = self.gac.models[model_idx]
                tokenizer = self.gac.tokenizers[model_idx]
                device = model.device
                
                # CRITICAL: Check if tokenizers are the same
                # If using different tokenizers, we need to handle this differently
                if model_idx > 0 and tokenizer != self.tokenizer:
                    # Convert tokens using this model's tokenizer
                    prefix_text = self.tokenizer.decode(prefix_tokens)
                    target_text = self.tokenizer.decode([target_token])
                    
                    # Re-tokenize with this model's tokenizer
                    model_prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
                    model_target_tokens = tokenizer.encode(target_text, add_special_tokens=False)
                    
                    if not model_target_tokens:
                        if debug:
                            print(f"      Model {model_idx}: target tokenization failed")
                        model_probs.append((0.0, weights[model_idx]))
                        continue
                    
                    # If target is multiple tokens in this tokenizer, get joint probability
                    log_prob_sum = 0.0
                    for j, tok in enumerate(model_target_tokens):
                        ctx = model_prefix + model_target_tokens[:j]
                        log_prob_sum += self._get_single_model_token_log_prob(
                            model, tokenizer, ctx, tok, device
                        )
                    
                    prob = math.exp(log_prob_sum) if log_prob_sum > -100 else 0.0
                    model_probs.append((prob, weights[model_idx]))
                    
                else:
                    # Same tokenizer, use tokens directly
                    log_prob = self._get_single_model_token_log_prob(
                        model, tokenizer, prefix_tokens, target_token, device
                    )
                    
                    prob = math.exp(log_prob) if log_prob > -100 else 0.0
                    model_probs.append((prob, weights[model_idx]))
                    
                    # Get greedy prediction
                    if len(prefix_tokens) > 0:
                        with torch.no_grad():
                            input_ids = torch.tensor([prefix_tokens], device=device)
                            # Truncate if needed
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
                model_probs.append((0.0, weights[model_idx]))
        
        # Calculate weighted ensemble probability
        if not model_probs:
            return -float('inf'), False
        
        ensemble_prob = sum(prob * weight for prob, weight in model_probs)
        
        # Convert back to log space
        if ensemble_prob > 0:
            ensemble_log_prob = math.log(ensemble_prob)
        else:
            ensemble_log_prob = -float('inf')
        
        # Check if target is greedy
        is_greedy = False
        if all_greedy_preds:
            # Simple majority vote
            greedy_vote = {}
            for pred in all_greedy_preds:
                greedy_vote[pred] = greedy_vote.get(pred, 0) + 1
            most_common = max(greedy_vote, key=greedy_vote.get)
            is_greedy = (most_common == target_token)
        
        if debug:
            print(f"      Ensemble: prob={ensemble_prob:.6f}, log_prob={ensemble_log_prob:.4f}")
        
        return ensemble_log_prob, is_greedy

    def _get_single_model_token_log_prob(self, model, tokenizer, prefix_tokens, target_token, device):
        """
        Get log probability from a single model for a single token.
        """
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

    def generate_until(self, requests):
        """Generate text until a stop sequence is found."""
        res = []

        for request_idx, request in enumerate(requests):
            context, until = request.args

            # Convert context to string if needed
            if isinstance(context, str):
                context_str = context
            else:
                # Handle list or tensor
                if isinstance(context, torch.Tensor):
                    context = context.tolist()
                context_str = self.tokenizer.decode(context)

            # Detect special tasks
            is_nq = "Answer these questions:" in context_str or ("Q:" in context_str and "A:" in context_str)

            # Get max generation length
            max_gen_length = 64  # Default
            if hasattr(request, "max_gen_length"):
                max_gen_length = request.max_gen_length

            # Process stop sequences
            if isinstance(until, str):
                stop_sequences = [until]
            elif isinstance(until, list) and all(isinstance(item, str) for item in until):
                stop_sequences = until
            else:
                # Default stop sequences
                stop_sequences = ["\n\n", "Q:"] if is_nq else ["\n", "\n\n"]

            try:
                # Use standard generate method with error handling
                try:
                    generated_text = self.gac.generate(
                        prompt=context_str,
                        max_length=max_gen_length,
                        temperature=0 if is_nq else 0.7,
                        top_p=0.9,
                        stop_sequences=stop_sequences,
                    )
                except Exception as e:
                    print(f"Error in primary generation method: {e}")
                    # Fallback to simpler method
                    generated_text = context_str + " [Generation failed]"

                # Get only the newly generated part
                generated_only = generated_text[len(context_str):]

                # For NQ, clean up the response
                if is_nq:
                    # Remove any repeated questions
                    if "Q:" in generated_only:
                        generated_only = generated_only.split("Q:")[0].strip()

                # IMPORTANT: For NQ, return the STRING directly, not tokens
                if is_nq:
                    res.append(generated_only)
                else:
                    # For other benchmarks, return tokens as before
                    generated_tokens = self.tokenizer.encode(
                        generated_only, add_special_tokens=False
                    )
                    res.append(generated_tokens)

            except Exception as e:
                print(f"Error in generate_until for request {request_idx}: {e}")
                traceback.print_exc()
                res.append("" if is_nq else [])
        
        return res

    def generate(self, context, max_gen_length, **kwargs):
        """Generate method for wrapper"""
        print(f"[Wrapper] Starting generation with max length {max_gen_length}...")
        if isinstance(context, str):
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
            context = context_tokens

        if isinstance(context, torch.Tensor):
            context = context.tolist()

        if (
            isinstance(context, list)
            and len(context) > 0
            and isinstance(context[0], torch.Tensor)
        ):
            context = [c.item() for c in context]

        # Decode context to string for GAC
        if isinstance(context, list):
            context_str = self.tokenizer.decode(context)
        else:
            context_str = context

        # Use GAC for generation
        temperature = kwargs.get("temperature", 0)  # Default to greedy decoding
        top_p = kwargs.get("top_p", 0.9)
        stop_tokens = kwargs.get("stop_tokens", ["\n", ".\n", "?\n", "!\n", "\n\n"])

        generated_text = self.gac.generate(
            prompt=context_str,
            max_length=max_gen_length,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_tokens,
        )

        # Get only the generated part (remove context)
        generated_only = generated_text[len(context_str):]

        # Return as list of token ids
        generated_tokens = self.tokenizer.encode(
            generated_only, add_special_tokens=False
        )
        return generated_tokens