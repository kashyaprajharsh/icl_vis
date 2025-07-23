from pydantic import BaseModel
from typing import List, Dict, Any

# --- Data Models for API ---

class GenerationRequest(BaseModel):
    """A request to the generation WebSocket endpoint."""
    model_name: str = "gpt2-medium"
    prompt: str
    max_length: int = 20
    temperature: float = 0.3

class Token(BaseModel):
    """Represents a single token with its ID and string representation."""
    id: int
    text: str

class TopPrediction(BaseModel):
    """Represents a single token prediction."""
    token: str
    probability: float

class InductionHeadDetail(BaseModel):
    """Details of a single induction head's performance."""
    layer: int
    head: int
    score: float

class CopyingHeadDetail(BaseModel):
    """Details of a single copying head's performance."""
    layer: int
    head: int
    score: float
    position: int
    copying_from: List[int]

class AttentionStats(BaseModel):
    """Statistics about the attention mechanism."""
    entropy: float
    max_attention: float
    sparsity: float
    diagonal_attention: float

class ICLMetrics(BaseModel):
    """Metrics related to in-context learning."""
    induction_score: float
    induction_heads: int
    copying_score: float
    copying_heads: int
    sequence_length: int
    generation_step: int
    induction_details: List[InductionHeadDetail]
    copying_details: List[CopyingHeadDetail] = []

class PatternStrength(BaseModel):
    """A summary of the strength of different attention patterns."""
    induction: float
    copying: float
    previous_token: float

class FinalAnalysis(BaseModel):
    """Final analysis after the generation is complete."""
    max_induction_score: float
    avg_induction_score: float
    pattern_diversity: int
    sequence_complexity: float

class PatternEvolutionPoint(BaseModel):
    """A single data point for the pattern evolution timeline."""
    step: int
    induction_score: float
    num_induction_heads: int
    induction_strength: float
    copying_strength: float
    previous_token_strength: float

class StepOutput(BaseModel):
    """The complete output for a single generation step."""
    step: int
    current_text: str
    tokens: List[Token]
    top_predictions: List[TopPrediction]
    icl_metrics: ICLMetrics
    attention_stats: AttentionStats
    attention_heatmap: Dict[str, Any]  # Changed from List[List[float]] to Any
    attention_graph: Dict[str, Any]
    token_importance: List[Dict[str, Any]]
    pattern_evolution: List[PatternEvolutionPoint]
    final_analysis: FinalAnalysis
    pattern_strength: PatternStrength
