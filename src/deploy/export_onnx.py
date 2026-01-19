import argparse
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def export_to_onnx(model_path, output_path, quantize=True):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Wrap model to return a single tensor (logits) instead of SequenceClassifierOutput
    class ExportModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped_model = ExportModel(model)

    # Create dummy input for tracing
    dummy_text = "This is a sample input for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    # Define input/output names
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    
    # Export
    print(f"Exporting to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        wrapped_model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )
    print("Export complete.")

    if quantize:
        print("Applying Int8 Dynamic Quantization...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_output_path = output_path.replace(".onnx", ".quant.onnx")
        quantize_dynamic(
            output_path,
            quantized_output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized model saved to {quantized_output_path}")
        return quantized_output_path
    
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save ONNX file")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)

if __name__ == "__main__":
    main()
