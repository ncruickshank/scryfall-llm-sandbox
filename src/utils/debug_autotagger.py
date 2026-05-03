import pandas as pd

def debug_autotagger_outputs(tagger, raw_dataset, num_samples=5):
    """
    Grabs samples from the validation set and compares ground truth to model output.
    """
    print(f"\n--- Model Output Debugging ({num_samples} Samples) ---")
    print(f'[GT = Ground Truth, PR = Model Prediction]')
    
    # Access the original validation data before it was tokenized/stripped
    # We need the 'document' for input and 'tags' for comparison
    val_data = raw_dataset['val']
    
    results = []
    
    for i in range(min(num_samples, len(val_data))):
        sample = val_data[i]
        card_text = sample['document']
        ground_truth = sample['tags']
        
        # Generate prediction using the model's current state
        # generate_tags internally uses self.model.generate()
        predicted_tags = tagger.generate_tags(card_text)
        
        # Format for display
        results.append({
            "Card Text": card_text[:100] + "...", 
            "Ground Truth": ground_truth,
            "Model Prediction": ", ".join(list(predicted_tags)) if predicted_tags else "[EMPTY]"
        })
    
    # Display results
    df = pd.DataFrame(results)
    for idx, row in df.iterrows():
        print(f"\nSample {idx+1}:")
        print(f"  GT: {row['Ground Truth']}")
        print(f"  PR: {row['Model Prediction']}")