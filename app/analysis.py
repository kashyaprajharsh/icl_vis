import torch
import numpy as np
from typing import List, Dict, Any

# --- Helper Functions for Analysis ---

def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Samples a token ID from the logits distribution scaled by temperature.
    """
    if temperature == 0 or temperature < 1e-8:
        # Use greedy decoding for temperature 0
        return torch.argmax(logits, dim=-1).unsqueeze(-1)
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply softmax to get probabilities
    probabilities = torch.softmax(scaled_logits, dim=-1)
    
    # Sample from the distribution
    sampled_token = torch.multinomial(probabilities, num_samples=1)
    
    return sampled_token

def normalize_attention_shape(attentions: np.ndarray) -> np.ndarray:
    """
    Normalize attention tensor to have shape (num_layers, num_heads, seq_len, seq_len).
    Handles different input shapes from different model architectures.
    """
    original_shape = attentions.shape
    
    if attentions.ndim == 2:
        # Shape: (seq_len, seq_len) - single head, single layer
        return attentions[np.newaxis, np.newaxis, ...]
    elif attentions.ndim == 3:
        # Shape: (num_heads, seq_len, seq_len) - single layer
        return attentions[np.newaxis, ...]
    elif attentions.ndim == 4:
        # Shape: (num_layers, num_heads, seq_len, seq_len) - already correct
        return attentions
    elif attentions.ndim == 5:
        # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
        # Remove batch dimension (assuming batch_size=1)
        if attentions.shape[1] == 1:
            return attentions.squeeze(1)  # Remove batch dimension
        else:
            # If batch_size > 1, take the first batch
            return attentions[:, 0, :, :, :]
    else:
        raise ValueError(f"Unexpected attention tensor shape: {original_shape}. "
                        f"Expected 2-5 dimensions, got {attentions.ndim}")


def calculate_attention_stats(attention_matrix: np.ndarray, seq_len: int) -> Dict[str, float]:
    """Calculates statistics for a given attention matrix."""
    if attention_matrix.size == 0:
        return {"entropy": 0.0, "max_attention": 0.0, "sparsity": 0.0}

    # Normalize shape
    attention_matrix = normalize_attention_shape(attention_matrix)
    
    # Slice the matrix to the actual sequence length to avoid padding issues
    valid_attention = attention_matrix[..., :seq_len, :seq_len]
    
    if valid_attention.size == 0:
        return {"entropy": 0.0, "max_attention": 0.0, "sparsity": 0.0}
    
    # Normalize attention matrices to ensure they sum to 1 along the last dimension
    valid_attention = valid_attention / (valid_attention.sum(axis=-1, keepdims=True) + 1e-9)
    
    # Calculate entropy (in bits) for each attention distribution
    entropy_per_position = -np.sum(valid_attention * np.log2(valid_attention + 1e-9), axis=-1)
    entropy = np.mean(entropy_per_position)
    
    # Calculate max attention
    max_attention = np.max(valid_attention, axis=-1).mean()
    
    # FIXED: Sparsity calculation (Hoyer sparsity measure)
    # Apply sparsity calculation row-wise for each attention distribution
    sparsity_scores = []
    for layer_idx in range(valid_attention.shape[0]):
        for head_idx in range(valid_attention.shape[1]):
            for pos_idx in range(valid_attention.shape[2]):
                attention_row = valid_attention[layer_idx, head_idx, pos_idx, :]
                l1_norm = np.linalg.norm(attention_row, ord=1)
                l2_norm = np.linalg.norm(attention_row, ord=2)
                
                if seq_len <= 1 or l2_norm < 1e-9:
                    sparsity_scores.append(0.0)
                else:
                    # Hoyer sparsity formula
                    hoyer_sparsity = (np.sqrt(seq_len) - l1_norm / l2_norm) / (np.sqrt(seq_len) - 1)
                    sparsity_scores.append(np.clip(hoyer_sparsity, 0, 1))
    
    sparsity = np.mean(sparsity_scores) if sparsity_scores else 0.0
    
    return {
        "entropy": float(entropy),
        "max_attention": float(max_attention),
        "sparsity": float(sparsity)
    }


def calculate_induction_score(tokens: List[int], attentions: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Calculates induction score following the exact mathematical definition from:
    "How Transformers Implement Induction Heads: Approximation and Optimization Analysis" (Wang et al.)
    
    Mathematical Definition: IH₂(X_L) = ∑_{s=2}^{L-1} softmax(x_L^T W* x_{s-1}) x_s
    
    Where:
    - x_L is the current (last) token
    - x_{s-1} are previous tokens we compare similarity against  
    - x_s are tokens we copy (that followed the similar previous tokens)
    - Pattern: [... A B ... A ?] → find previous A, copy following B
    """
    if len(tokens) < 3:
        return {"score": 0.0, "count": 0, "details": []}

    try:
        attentions = normalize_attention_shape(attentions)
    except ValueError as e:
        print(f"Warning: Could not normalize attention shape: {e}")
        return {"score": 0.0, "count": 0, "details": []}
    
    total_score = 0.0
    induction_head_count = 0
    induction_details = []
    num_layers, num_heads, seq_len_full, _ = attentions.shape
    
    actual_seq_len = min(len(tokens), seq_len_full)
    if actual_seq_len < 3:
        return {"score": 0.0, "count": 0, "details": []}
    
    attentions = attentions[:, :, :actual_seq_len, :actual_seq_len]
    
    # x_L = current token (last token in sequence)
    x_L = tokens[-1]
    L = len(tokens)
    
    # For each layer and head, calculate induction score
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            layer_head_score = 0.0
            matching_patterns = []
            
            # ∑_{s=2}^{L-1} softmax(x_L^T W* x_{s-1}) x_s
            # s ranges from 2 to L-1 (1-indexed in paper, 0-indexed in code: s from 1 to L-2)
            for s in range(1, L-1):  # s=1 to L-2 in 0-indexed
                x_s_minus_1 = tokens[s-1]  # x_{s-1}: token at position s-1 # Previous token
                x_s = tokens[s]            # x_s: token at position s (to be copied) # What to copy
                
                # Check similarity: does current token x_L match previous token x_{s-1}?
                if x_L == x_s_minus_1: #pattern match i.e prefix matching
                    # Get attention weight from current position (L-1) to position s
                    # This represents the softmax weight in the formula
                    attention_weight = attentions[layer_idx, head_idx, L-1, s]
                    
                    if attention_weight > threshold:
                        layer_head_score += float(attention_weight)
                        matching_patterns.append({
                            "similarity_pos": s-1,      # Position of x_{s-1} (similar token)
                            "copy_pos": s,              # Position of x_s (token to copy)
                            "similar_token": x_s_minus_1,
                            "copy_token": x_s,
                            "attention_weight": float(attention_weight)
                        })
            
            # If this head shows induction behavior above threshold
            if layer_head_score > threshold:
                total_score += layer_head_score
                induction_head_count += 1
                induction_details.append({
                    "layer": layer_idx,
                    "head": head_idx, 
                    "score": layer_head_score,
                    "matching_patterns": matching_patterns,
                    "formula_info": {
                        "current_token_x_L": x_L,
                        "sequence_length_L": L,
                        "pattern_description": f"Found {len(matching_patterns)} induction patterns"
                    }
                })
    
    return {
        "score": total_score, 
        "count": induction_head_count, 
        "details": induction_details
    }


def calculate_copying_score(tokens: List[int], attentions: np.ndarray, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Calculate copying score - how much the model copies from previous positions.
    This is complementary to induction and helps understand ICL mechanisms.
    """
    if len(tokens) < 2:
        return {"score": 0.0, "count": 0, "details": []}
    
    try:
        attentions = normalize_attention_shape(attentions)
    except ValueError:
        return {"score": 0.0, "count": 0, "details": []}
    
    num_layers, num_heads, seq_len, _ = attentions.shape
    actual_seq_len = min(len(tokens), seq_len)
    
    if actual_seq_len < 2:
        return {"score": 0.0, "count": 0, "details": []}
    
    attentions = attentions[:, :, :actual_seq_len, :actual_seq_len]
    
    total_score = 0.0
    copying_head_count = 0
    copying_details = []
    
    # Look at attention from the last token to previous positions with the same token ID
    last_token_id = tokens[-1]
    last_token_attention = attentions[:, :, -1, :-1] # Attention from last token to all previous
    
    # Find previous positions with the same token
    same_token_positions = [i for i, token_id in enumerate(tokens[:-1]) if token_id == last_token_id]
    
    if not same_token_positions:
        return {"score": 0.0, "count": 0, "details": []}
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            head_attention = last_token_attention[layer_idx, head_idx]
            
            # Sum attention to all previous positions with the same token
            copying_attention = sum(
                head_attention[prev_pos] 
                for prev_pos in same_token_positions if prev_pos < len(head_attention)
            )
            
            if copying_attention > threshold:
                total_score += float(copying_attention)
                copying_head_count += 1
                copying_details.append({
                    "layer": layer_idx,
                    "head": head_idx,
                    "score": float(copying_attention),
                    "position": actual_seq_len - 1,
                    "copying_from": same_token_positions
                })
    
    return {
        "score": total_score,
        "count": copying_head_count,
        "details": copying_details
    }


def create_attention_graph(attention_matrix: np.ndarray, tokens: List[str], threshold: float = 0.1) -> Dict[str, Any]:
    """
    Creates a directed graph from an attention matrix.
    The matrix should be (seq_len, seq_len) or will be averaged to that shape.
    """
    # Handle different attention shapes
    if attention_matrix.ndim > 2:
        # Average over layers and heads to get (seq_len, seq_len)
        while attention_matrix.ndim > 2:
            attention_matrix = attention_matrix.mean(axis=0)
    
    if attention_matrix.ndim != 2:
        raise ValueError(f"Could not reduce attention matrix to 2D. Current shape: {attention_matrix.shape}")
    
    seq_len = min(len(tokens), attention_matrix.shape[0])
    
    # Clean token text for display
    clean_tokens = [tok.replace('<', '&lt;').replace('>', '&gt;').replace('\n', '\\n') for tok in tokens[:seq_len]]
    
    nodes = [{'id': i, 'label': f"{clean_tokens[i]} ({i})", 'position': i} for i in range(seq_len)]
    edges = []

    for i in range(seq_len):  # To token (Query)
        for j in range(seq_len):  # From token (Key)
            if i < attention_matrix.shape[0] and j < attention_matrix.shape[1]:
                weight = attention_matrix[i, j]
                if weight > threshold:
                    edges.append({
                        'from': j, 
                        'to': i, 
                        'value': float(weight), 
                        'arrows': 'to'
                    })
    
    return {'nodes': nodes, 'edges': edges}


def calculate_previous_token_attention(attentions: np.ndarray) -> float:
    """
    Calculates the average attention paid to the token immediately preceding the current one.
    This is a common positional attention pattern.
    """
    if attentions.size == 0:
        return 0.0

    try:
        attentions = normalize_attention_shape(attentions)
    except ValueError:
        return 0.0

    num_layers, num_heads, seq_len, _ = attentions.shape
    if seq_len < 2:
        return 0.0

    # Attention from the last token to the second-to-last token
    # Shape: (num_layers, num_heads)
    prev_token_attention = attentions[:, :, -1, -2]

    # Average this score across all heads and layers
    return float(np.mean(prev_token_attention))


def calculate_token_importance(attentions: np.ndarray, tokens: List[str]) -> List[Dict[str, Any]]:
    """
    Analyzes the importance of each token in the sequence.
    """
    if attentions.size == 0 or not tokens:
        return []

    # Normalize attention shape
    try:
        attentions = normalize_attention_shape(attentions)
    except ValueError as e:
        print(f"Warning: Could not normalize attention shape: {e}")
        return []

    actual_seq_len = len(tokens)
    
    # Average attention across layers and heads
    avg_attention = attentions[..., :actual_seq_len, :actual_seq_len].mean(axis=(0, 1))
    
    token_importance = []
    for i in range(actual_seq_len):
        # Incoming attention (how much other tokens attend to this token)
        incoming = avg_attention[:, i].sum()
        
        # Outgoing attention (how much this token attends to others)
        outgoing = avg_attention[i, :].sum()
        
        # FIXED: Attention entropy (how focused/dispersed the attention is)
        attention_row = avg_attention[i, :]
        # Normalize to get probability distribution
        attention_dist = attention_row / (attention_row.sum() + 1e-9)
        entropy = -np.sum(attention_dist * np.log2(attention_dist + 1e-9))

        token_importance.append({
            "token": tokens[i],
            "position": i,
            "incoming_attention": float(incoming),
            "outgoing_attention": float(outgoing),
            "attention_entropy": float(entropy)
        })
    
    return token_importance


def calculate_diagonal_attention(attention_matrix: np.ndarray) -> float:
    """Calculate how much attention is on the diagonal (self-attention)"""
    if attention_matrix.ndim != 2:
        return 0.0
    
    n = min(attention_matrix.shape)
    if n <= 1:
        return 0.0
    
    diagonal_sum = np.trace(attention_matrix)
    return float(diagonal_sum / n)
