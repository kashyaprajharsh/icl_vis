import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logger = logging.getLogger(__name__)

# Global cache to avoid reloading models
_model_cache = {}

def get_or_load_model(model_name: str):
    """Load and cache a model and tokenizer."""
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,  # Important for some models
            output_attentions=True,  # Ensure attention outputs are enabled
        )
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            model = model.to(device)
        
        model.eval()  # Set to evaluation mode
        
        model_data = {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
        
        _model_cache[model_name] = model_data
        logger.info(f"Model {model_name} loaded successfully on {device}")
        return model_data
        
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise