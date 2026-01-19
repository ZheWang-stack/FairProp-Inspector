import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(text, model_path):
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"⚠️  Could not find trained model at {model_path}. Loading base model 'distilbert-base-uncased' (untrained on this task).")
        model_path = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    
    # If the model has config with labels, use them
    if hasattr(model.config, "id2label") and model.config.id2label:
        label = model.config.id2label[predicted_class_id]
    else:
        label = "NON_COMPLIANT" if predicted_class_id == 1 else "COMPLIANT" # Fallback assumption
        
    confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
    
    return label, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text to classify")
    parser.add_argument("--model", type=str, default="artifacts/model", help="Path to saved model")
    args = parser.parse_args()

    label, confidence = predict(args.text, args.model)
    
    print("-" * 30)
    print(f"Input:      {args.text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 30)

if __name__ == "__main__":
    main()
