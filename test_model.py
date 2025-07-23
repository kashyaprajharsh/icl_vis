from app.model_loader import get_or_load_model

def test_model_generation():
    """
    A test script to directly invoke a model and check its generation output.
    """
    model_name = "Qwen/Qwen3-0.6B"
    # The prompt from the UI screenshot
    prompt = "English: Hello -> French: Bonjour\nEnglish: Thank you -> French: Merci\nEnglish: Good morning -> French:"

    print(f"--- Testing Model: {model_name} ---")
    print(f"Prompt:\n{prompt}\n")

    try:
        # 1. Load model and tokenizer
        model_data = get_or_load_model(model_name)
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        print("Model and tokenizer loaded successfully.")

        # 2. Encode the prompt
        token_ids = tokenizer.encode(prompt, return_tensors="pt")
        print(f"Input token IDs: {token_ids}")

        # 3. Generate output
        print("\nGenerating output...")
        # Using model.generate for a simple end-to-end test
        output_ids = model.generate(
            token_ids,
            max_new_tokens=10,
            temperature=0.43,
            pad_token_id=tokenizer.eos_token_id # Set pad_token_id to suppress warning
        )
        print(f"Output token IDs: {output_ids}")

        # 4. Decode and print the full output
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("\n--- Full Generated Text ---")
        print(full_text)

        # 5. Decode and print only the newly generated part
        generated_ids = output_ids[0][len(token_ids[0]):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("\n--- Newly Generated Text ---")
        print(generated_text)
        print("\n--- Test Complete ---")

    except Exception:
        print("\n--- An error occurred ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_generation()
