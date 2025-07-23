from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import numpy as np
import logging
import asyncio

from .models import (
    GenerationRequest,
    Token,
    TopPrediction,
    AttentionStats,
    ICLMetrics,
    FinalAnalysis,
    PatternEvolutionPoint,
    StepOutput,
    PatternStrength,
)
from .utils import ConnectionManager
from .model_loader import get_or_load_model
from .analysis import (
    temperature_sampling,
    calculate_attention_stats,
    calculate_induction_score,
    calculate_copying_score,
    create_attention_graph,
    calculate_token_importance,
    calculate_diagonal_attention,
    normalize_attention_shape,
    calculate_previous_token_attention,
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App Setup ---
app = FastAPI()
manager = ConnectionManager()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # --- Mount Static Files and Templates ---
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# --- Sample ICL Patterns Library ---
SAMPLE_ICL_PATTERNS = {
    "translation": {
        "name": "Translation",
        "prompt": "English -> French\nHello -> Bonjour\nGoodbye -> Au revoir\nThank you -> "
    },
    "arithmetic": {
        "name": "Arithmetic",
        "prompt": "1 + 1 = 2\n2 + 2 = 4\n3 + 3 = 6\n4 + 4 ="
    },
    "repetition": {
        "name": "Repetition",
        "prompt": "The cat says meow. The dog says woof. The cat says meow. The pig says"
    },
    "qa": {
        "name": "Question Answering",
        "prompt": "Question: What is the capital of France?\nAnswer: Paris\nQuestion: What is the capital of Germany?\nAnswer: Berlin\nQuestion: What is the capital of Spain?\nAnswer:"
    },
    "sequence": {
        "name": "Numeric Sequence",
        "prompt": "1, 2, 3, 4, 5\n10, 20, 30, 40, 50\n5, 10, 15, 20,"
    },
    "summarization": {
        "name": "Summarization",
        "prompt": "Text: Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined. Summary: Jupiter is the largest planet in the Solar System.\nText: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Summary:"
    },
    "sentiment": {
        "name": "Sentiment Analysis",
        "prompt": "Sentence: I love this movie, it's fantastic! Sentiment: Positive\nSentence: I'm not sure how I feel about this. Sentiment: Neutral\nSentence: This was a complete waste of time. Sentiment:"
    },
    "analogy": {
        "name": "Analogy",
        "prompt": "apple is to fruit as carrot is to vegetable\ncat is to kitten as dog is to puppy\nsky is to blue as grass is to"
    },
    "python_code": {
        "name": "Python Code Generation",
        "prompt": "# Python function to add two numbers\ndef add(a, b):\n    return a + b\n\n# Python function to subtract two numbers\ndef subtract(a, b):\n    return a - b\n\n# Python function to multiply two numbers\n"
    }
}

@app.get("/")
async def read_root():
    """Serves a welcome message."""
    return {"message": "Welcome to the GPT-2 Visualization API"}

@app.get("/sample-patterns")
def get_sample_patterns():
    """Return sample ICL patterns for the frontend"""
    return {"patterns": SAMPLE_ICL_PATTERNS}

# --- Enhanced WebSocket Endpoint ---
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connection accepted.")
    
    try:
        while True:
            request_json = await websocket.receive_json()
            request = GenerationRequest(**request_json)
            logger.info(f"Received generation request for model '{request.model_name}' with prompt: '{request.prompt[:50]}...'")

            # --- Load Model ---
            model_data = get_or_load_model(request.model_name)
            tokenizer, model = model_data["tokenizer"], model_data["model"]
            device = model.device
            
            # --- Initialize Generation State ---
            # Encode the prompt
            input_ids = tokenizer.encode(request.prompt, return_tensors="pt").to(device)
            logger.info(f"Input tokens: {input_ids}")
            logger.info(f"Input text: '{tokenizer.decode(input_ids[0])}'")
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids).to(device)
            
            # Initialize past key values
            past_key_values = None
            
            pattern_evolution_data = []
            final_analysis_metrics = {
                "max_induction_score": 0.0,
                "avg_induction_score": 0.0,
                "total_induction_score": 0.0,
                "pattern_diversity": 0,
                "sequence_complexity": 0.0,
                "detected_patterns": {}
            }

            for step in range(request.max_length):
                logger.info(f"Generation step {step}")
                
                # --- Generate Next Token ---
                with torch.no_grad():
                    # Prepare inputs for generation. When using past_key_values (use_cache=True),
                    # we only need to pass the last token and the full attention mask.
                    if past_key_values is not None:
                        # For subsequent steps, only use the last token
                        model_input_ids = input_ids[:, -1:]
                    else:
                        # For the first step, use the full sequence
                        model_input_ids = input_ids
                    
                    # Generate
                    outputs = model(
                        input_ids=model_input_ids,
                        attention_mask=attention_mask, # Always use the full attention mask
                        past_key_values=past_key_values,
                        output_attentions=True,
                        use_cache=True,
                        return_dict=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # Sample next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token_id = temperature_sampling(next_token_logits, request.temperature)

                # For analysis, get the probabilities
                if request.temperature > 1e-8:
                    scaled_logits = next_token_logits / request.temperature
                    next_token_probs = torch.softmax(scaled_logits, dim=-1)
                else:
                    # For temperature 0, the distribution is a one-hot vector
                    # at the position of the most likely token.
                    next_token_probs = torch.zeros_like(next_token_logits)
                    next_token_probs.scatter_(1, next_token_id, 1.0)
                
                logger.info(f"Generated token ID: {next_token_id.item()}")
                logger.info(f"Generated token: '{tokenizer.decode(next_token_id[0])}'")
                
                # Update sequences
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=device)
                ], dim=1)
                
                # Update past_key_values for next iteration
                past_key_values = outputs.past_key_values

                # --- Decode current sequence ---
                current_tokens_list = input_ids.squeeze(0).tolist()
                decoded_tokens = [tokenizer.decode([t], skip_special_tokens=True) for t in current_tokens_list]
                full_text = tokenizer.decode(current_tokens_list, skip_special_tokens=True)
                
                logger.info(f"Current text: '{full_text}'")
                
                # --- Get full attention for analysis ---
                with torch.no_grad():
                    # We need full attention matrix for analysis
                    full_outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        return_dict=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                # Process attention tensors
                try:
                    if hasattr(full_outputs, 'attentions') and full_outputs.attentions:
                        # Convert to numpy and normalize
                        all_attentions = torch.stack(full_outputs.attentions).cpu().numpy()
                        all_attentions = normalize_attention_shape(all_attentions)
                        logger.info(f"Attention shape: {all_attentions.shape}")
                    else:
                        logger.warning("No attention outputs found")
                        seq_len = len(current_tokens_list)
                        all_attentions = np.ones((1, 1, seq_len, seq_len)) / seq_len
                        
                except Exception as e:
                    logger.error(f"Error processing attention: {e}")
                    seq_len = len(current_tokens_list)
                    all_attentions = np.ones((1, 1, seq_len, seq_len)) / seq_len
                
                # ICL Metrics
                try:
                    induction_results = calculate_induction_score(current_tokens_list, all_attentions)
                    copying_results = calculate_copying_score(current_tokens_list, all_attentions)
                    icl_metrics = ICLMetrics(
                        induction_score=induction_results["score"],
                        induction_heads=induction_results["count"],
                        copying_score=copying_results["score"],
                        copying_heads=copying_results["count"],
                        induction_details=induction_results["details"],
                        copying_details=copying_results["details"],
                        sequence_length=len(current_tokens_list),
                        generation_step=step
                    )
                except Exception as e:
                    logger.error(f"Error calculating ICL metrics: {e}")
                    icl_metrics = ICLMetrics(
                        induction_score=0.0,
                        induction_heads=0,
                        copying_score=0.0,
                        copying_heads=0,
                        induction_details=[],
                        sequence_length=len(current_tokens_list),
                        generation_step=step
                    )

                # Attention Statistics
                try:
                    attention_stats_results = calculate_attention_stats(all_attentions, len(current_tokens_list))
                    # FIXED: Calculate diagonal attention per head, then average
                    diagonal_attention_scores = []
                    for layer_att in all_attentions:
                        for head_idx in range(layer_att.shape[0]):
                            head_diagonal = calculate_diagonal_attention(layer_att[head_idx])
                            diagonal_attention_scores.append(head_diagonal)
                    diagonal_attention = np.mean(diagonal_attention_scores) if diagonal_attention_scores else 0.0
                    
                    attention_stats = AttentionStats(
                        **attention_stats_results,
                        diagonal_attention=diagonal_attention
                    )
                except Exception as e:
                    logger.error(f"Error calculating attention stats: {e}")
                    attention_stats = AttentionStats(
                        entropy=0.0,
                        max_attention=0.0,
                        sparsity=0.0,
                        diagonal_attention=0.0
                    )

                # Top Predictions
                top_k = torch.topk(next_token_probs, 5)
                top_predictions = [
                    TopPrediction(
                        token=tokenizer.decode([idx.item()], skip_special_tokens=True),
                        probability=prob.item()
                    )
                    for idx, prob in zip(top_k.indices.squeeze(), top_k.values.squeeze())
                ]

                # Attention Heatmap
                try:
                    attention_heatmap = {}
                    for i, layer_att in enumerate(all_attentions):
                        layer_avg = layer_att.mean(axis=0)
                        attention_heatmap[f"layer_{i}"] = layer_avg.tolist()
                except Exception as e:
                    logger.error(f"Error creating attention heatmap: {e}")
                    attention_heatmap = {"layer_0": [[0.0]]}

                # Attention Graph
                try:
                    last_layer_avg_attention = all_attentions[-1].mean(axis=0)
                    attention_graph = create_attention_graph(last_layer_avg_attention, decoded_tokens)
                except Exception as e:
                    logger.error(f"Error creating attention graph: {e}")
                    attention_graph = {'nodes': [], 'edges': []}

                # Token Importance
                try:
                    token_importance = calculate_token_importance(all_attentions, decoded_tokens)
                except Exception as e:
                    logger.error(f"Error calculating token importance: {e}")
                    token_importance = []

                # Pattern Strength Summary
                num_heads = model.config.num_attention_heads
                num_layers = model.config.num_hidden_layers
                total_heads = num_heads * num_layers
                
                prev_token_strength = calculate_previous_token_attention(all_attentions)

                # FIXED: Better normalization based on theoretical maximums
                # Each head can contribute at most 1.0 attention to induction/copying
                # So normalize by number of active heads, not arbitrary values
                induction_strength = 0.0
                if icl_metrics.induction_heads > 0:
                    # Average attention per induction head (0 to 1)
                    induction_strength = min(1.0, icl_metrics.induction_score / icl_metrics.induction_heads)
                
                copying_strength = 0.0
                if icl_metrics.copying_heads > 0:
                    # Average attention per copying head (0 to 1)
                    copying_strength = min(1.0, icl_metrics.copying_score / icl_metrics.copying_heads)

                pattern_strength = PatternStrength(
                    induction=induction_strength,
                    copying=copying_strength,
                    previous_token=prev_token_strength
                )

                # Pattern Evolution
                pattern_evolution_data.append(PatternEvolutionPoint(
                    step=step,
                    induction_score=icl_metrics.induction_score,
                    num_induction_heads=icl_metrics.induction_heads,
                    induction_strength=pattern_strength.induction,
                    copying_strength=pattern_strength.copying,
                    previous_token_strength=pattern_strength.previous_token
                ))
                
                # Final Analysis
                final_analysis_metrics["total_induction_score"] += icl_metrics.induction_score
                final_analysis_metrics["max_induction_score"] = max(
                    final_analysis_metrics["max_induction_score"], 
                    icl_metrics.induction_score
                )
                final_analysis_metrics["avg_induction_score"] = (
                    final_analysis_metrics["total_induction_score"] / (step + 1)
                )
                # FIXED: Better sequence complexity metric based on attention entropy
                avg_entropy = attention_stats.entropy if attention_stats.entropy > 0 else 1.0
                sequence_length_factor = np.log(step + 2)
                final_analysis_metrics["sequence_complexity"] = avg_entropy * sequence_length_factor
                final_analysis_metrics["pattern_diversity"] = 0 # Set to 0 as detected_patterns is removed
                # final_analysis_metrics["pattern_diversity"] = len([
                #     v for v in final_analysis_metrics["detected_patterns"].values() if v > 0
                # ])

                final_analysis = FinalAnalysis(**final_analysis_metrics)

                # --- Assemble Step Output ---
                step_output = StepOutput(
                    step=step,
                    current_text=full_text,
                    tokens=[Token(id=tid, text=tok) for tid, tok in zip(current_tokens_list, decoded_tokens)],
                    top_predictions=top_predictions,
                    icl_metrics=icl_metrics,
                    attention_stats=attention_stats,
                    attention_heatmap=attention_heatmap,
                    attention_graph=attention_graph,
                    token_importance=token_importance,
                    pattern_evolution=pattern_evolution_data,
                    final_analysis=final_analysis,
                    pattern_strength=pattern_strength
                )

                await manager.send_json(step_output.dict(), websocket)
                await asyncio.sleep(0.1)
                
                # Check for stopping conditions
                if next_token_id.item() == tokenizer.eos_token_id:
                    logger.info("EOS token generated, stopping")
                    break
                
                # Check for pad token
                if next_token_id.item() == tokenizer.pad_token_id:
                    logger.info("Pad token generated, stopping")
                    break

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected.")
    except Exception as e:
        logger.exception("An unexpected error occurred in the websocket handler.")
        try:
            await manager.send_json({"type": "error", "message": str(e)}, websocket)
        except Exception:
            pass
