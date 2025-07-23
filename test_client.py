import asyncio
import websockets
import json

async def test_generate():
    uri = "ws://localhost:8000/ws/generate"
    async with websockets.connect(uri) as websocket:
        # Sample request based on the new features
        request_data = {
            "model_name": "gpt2",
            "prompt": "1 + 1 = 2\n2 + 2 = 4\n3 + 3 =",
            "max_length": 15,
            "temperature": 0.1
        }
        
        await websocket.send(json.dumps(request_data))
        print(f"> Sent request: {request_data['prompt']}")

        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get("type") == "error":
                    print(f"! Error from server: {data['message']}")
                    break
                
                print("\n--- New Step Received ---")
                print(f"Step: {data['step']}")
                print(f"Current Text: {data['current_text']}")
                
                # Print ICL Metrics
                icl_metrics = data['icl_metrics']
                print("\nICL Metrics:")
                print(f"  Induction Score: {icl_metrics['induction_score']:.4f}")
                print(f"  Induction Heads: {icl_metrics['induction_heads']}")
                print(f"  Sequence Length: {icl_metrics['sequence_length']}")
                
                # Print Attention Stats
                attn_stats = data['attention_stats']
                print("\nAttention Stats:")
                print(f"  Entropy: {attn_stats['entropy']:.4f}")
                print(f"  Max Attention: {attn_stats['max_attention']:.4f}")
                print(f"  Sparsity: {attn_stats['sparsity']:.4f}")

                # Print Top Predictions
                print("\nTop Predictions:")
                for pred in data['top_predictions']:
                    print(f"  - Token: '{pred['token']}', Probability: {pred['probability']:.4f}")
                
                if data['step'] == request_data['max_length'] - 1:
                    print("\n--- Final Analysis ---")
                    final_analysis = data['final_analysis']
                    print(f"Max Induction Score: {final_analysis['max_induction_score']:.4f}")
                    print(f"Avg Induction Score: {final_analysis['avg_induction_score']:.4f}")
                    print(f"Detected Patterns: {final_analysis['detected_patterns']}")
                    break

        except websockets.exceptions.ConnectionClosed:
            print("\nConnection closed by the server.")

if __name__ == "__main__":
    asyncio.run(test_generate())