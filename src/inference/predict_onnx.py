import argparse
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def predict_onnx(text, model_dir):
    print(f"🔄 Loading ONNX model from {model_dir}...")
    
    # Path to model and tokenizer
    model_path = f"{model_dir}/model.onnx"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Initialize ONNX Runtime session
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Preprocess text
    inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=512)
    
    # Prepare ONNX inputs
    # session.get_inputs() gives you the names: 'input_ids' and 'attention_mask'
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    
    # Run inference
    outputs = session.run(None, onnx_inputs)
    logits = outputs[0]
    
    # Get prediction
    predicted_class_id = np.argmax(logits)
    
    # Get labels from tokenizer config if available
    id2label = {0: "COMPLIANT", 1: "NON_COMPLIANT"}
    label = id2label[predicted_class_id]
    
    # Calculate confidence (Softmax)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    confidence = probs[0][predicted_class_id]
    
    return label, confidence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Text to classify")
    parser.add_argument("--model_dir", type=str, default="artifacts/model_onnx", help="Directory containing model.onnx and tokenizer files")
    args = parser.parse_args()

    label, confidence = predict_onnx(args.text, args.model_dir)
    
    print("-" * 40)
    print(f"Input:      {args.text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()
