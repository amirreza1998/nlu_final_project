import torch
import json
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import gc
import os
from typing import Dict, List, Tuple
import re

# Improved configuration
class Config:
    HF_TOKEN = "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"
    MODEL_NAME = "CohereLabs/aya-23-8B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH = 512
    OUTPUT_DIR = "./medical_lora_output"
    FINAL_MODEL_DIR = "./final_medical_lora"

def load_base_model():
    """Enhanced model loading with better error handling"""
    print(f"بارگذاری مدل {Config.MODEL_NAME}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            trust_remote_code=True, 
            token=Config.HF_TOKEN,
            use_fast=True  # Use fast tokenizer for better performance
        )
        
        # Better padding configuration
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Optimized quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=torch.uint8  # Better storage efficiency
        )

        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=Config.HF_TOKEN,
            torch_dtype=torch.bfloat16,  # Consistent dtype
            low_cpu_mem_usage=True  # Better memory management
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return model, tokenizer
    
    except Exception as e:
        print(f"خطا در بارگذاری مدل: {e}")
        raise

def prepare_training_data_enhanced(data_path: str, tokenizer, validation_split: float = 0.15):
    """Enhanced data preparation with better validation split and formatting"""
    
    print("آماده‌سازی داده‌های آموزشی...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    utterances = data.get("Utterance", [])
    formatted_data = []
    labels = []  # For evaluation purposes

    for item in utterances:
        query = item.get("query", "")
        diseases = item.get("diseases", [])
        
        if not query or not diseases:
            continue
            
        diseases_text = "، ".join(diseases)
        
        # Enhanced prompt template with better structure
        formatted_text = f"""<|system|>
شما یک دستیار پزشکی متخصص هستید که بر اساس علائم ارائه شده، بیماری‌های محتمل را تشخیص می‌دهید. پاسخ خود را به صورت دقیق و علمی ارائه دهید.

<|user|>
{query}

<|assistant|>
با توجه به علائم ذکر شده، بیماری‌های محتمل عبارتند از: {diseases_text}

برای تشخیص دقیق‌تر، توصیه می‌شود با پزشک متخصص مشورت کنید و آزمایش‌های لازم را انجام دهید.<|end|>"""
        
        formatted_data.append({"text": formatted_text})
        labels.append(diseases)

    # Create dataset with labels for evaluation
    df = pd.DataFrame({"text": [item["text"] for item in formatted_data], "labels": labels})
    dataset = Dataset.from_pandas(df)

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=Config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Add labels column for evaluation
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset.column_names
    )

    # Better train/validation split
    split_dataset = tokenized_dataset.train_test_split(
        test_size=validation_split,
        seed=42,
        shuffle=True
    )

    return split_dataset

def create_optimized_lora_config():
    """Optimized LoRA configuration for better performance"""
    
    return LoraConfig(
        r=8,  # Increased rank for better capacity
        lora_alpha=32,  # Proper scaling
        lora_dropout=0.1,  # Slightly higher dropout for regularization
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"  # Include more modules for better adaptation
        ],
        modules_to_save=["embed_tokens", "lm_head"]  # Save important modules
    )

class MetricsCallback(TrainerCallback):
    """Enhanced callback for better monitoring"""
    
    def __init__(self):
        self.training_losses = []
        self.eval_losses = []
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            if "train_loss" in logs:
                self.training_losses.append(logs["train_loss"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
            
            # Memory monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"GPU Memory: {allocated:.2f}GB")

def train_model_enhanced(model, tokenizer, tokenized_dataset, lora_config):
    """Enhanced training with better monitoring and optimization"""
    
    print("پاک‌سازی حافظه...")
    torch.cuda.empty_cache()
    gc.collect()
    
    print("آماده‌سازی مدل LoRA...")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=5,  # Increased epochs
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        
        # Learning rate scheduling
        learning_rate=1e-4,  # Slightly lower learning rate
        lr_scheduler_type="cosine_with_restarts",
        warmup_steps=100,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        
        # Optimization
        fp16=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Logging
        logging_steps=50,
        logging_first_step=True,
        report_to="none",
        
        # Advanced optimizations
        optim="adamw_torch_fused",  # Faster optimizer
        max_grad_norm=1.0,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        
        # Memory optimizations
        include_inputs_for_metrics=False,
        prediction_loss_only=False,  # Enable for metrics calculation
    )

    # Enhanced data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    # Callbacks
    metrics_callback = MetricsCallback()

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[metrics_callback]
    )

    print("شروع آموزش...")
    
    try:
        training_result = trainer.train()
        print(f"آموزش کامل شد. Final loss: {training_result.training_loss:.4f}")
        
        # Save model with better configuration
        peft_model.save_pretrained(
            Config.FINAL_MODEL_DIR,
            safe_serialization=True,
            max_shard_size="1GB"
        )
        tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)
        
        # Save training history
        history = {
            "training_losses": metrics_callback.training_losses,
            "eval_losses": metrics_callback.eval_losses,
            "final_loss": training_result.training_loss
        }
        
        with open(f"{Config.FINAL_MODEL_DIR}/training_history.json", "w") as f:
            json.dump(history, f, indent=2)
            
        return peft_model, history
        
    except Exception as e:
        print(f"خطا در آموزش: {e}")
        raise

def evaluate_model_comprehensive(model_path: str, test_data_path: str = None):
    """Comprehensive model evaluation with multiple metrics"""
    
    print("ارزیابی جامع مدل...")
    
    # Load model and tokenizer
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    # Test questions with expected answers for evaluation
    test_cases = [
        {
            "query": "من سردرد شدید، تب و گلودرد دارم",
            "expected_diseases": ["آنفولانزا", "سرماخوردگی", "التهاب گلو"],
            "category": "respiratory"
        },
        {
            "query": "درد قفسه سینه، تنگی نفس و خستگی دارم",
            "expected_diseases": ["بیماری قلبی", "آنژین صدری", "نارسایی قلبی"],
            "category": "cardiovascular"
        },
        {
            "query": "درد شکم، تهوع و اسهال دارم",
            "expected_diseases": ["گاستروانتریت", "عفونت روده", "مسمومیت غذایی"],
            "category": "gastrointestinal"
        },
        {
            "query": "سردرد مداوم، تاری دید و سرگیجه",
            "expected_diseases": ["فشار خون بالا", "میگرن", "مشکلات بینایی"],
            "category": "neurological"
        },
        {
            "query": "درد مفاصل، خشکی دهان و خستگی",
            "expected_diseases": ["آرتریت روماتوئید", "لوپوس", "سندرم خشکی"],
            "category": "autoimmune"
        }
    ]

    results = {
        "accuracy_scores": [],
        "precision_scores": [],
        "recall_scores": [],
        "f1_scores": [],
        "category_performance": {},
        "detailed_results": []
    }

    for i, test_case in enumerate(test_cases):
        print(f"\nارزیابی مورد {i+1}/{len(test_cases)}: {test_case['category']}")
        
        prompt = f"""<|system|>
شما یک دستیار پزشکی متخصص هستید که بر اساس علائم ارائه شده، بیماری‌های محتمل را تشخیص می‌دهید.

<|user|>
{test_case['query']}

<|assistant|>
با توجه به علائم ذکر شده، بیماری‌های محتمل عبارتند از:"""

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent results
                do_sample=True,
                top_p=0.8,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response.split("<|assistant|>")[-1].strip()
        
        # Extract predicted diseases using regex
        disease_pattern = r'(?:بیماری‌های محتمل|عبارتند از)[:\s]*([^\.]+)'
        match = re.search(disease_pattern, generated_text)
        
        if match:
            predicted_diseases_text = match.group(1)
            predicted_diseases = [disease.strip() for disease in predicted_diseases_text.split('،')]
        else:
            predicted_diseases = []

        # Calculate metrics
        accuracy = calculate_disease_accuracy(predicted_diseases, test_case['expected_diseases'])
        precision, recall, f1 = calculate_disease_metrics(predicted_diseases, test_case['expected_diseases'])
        
        results["accuracy_scores"].append(accuracy)
        results["precision_scores"].append(precision)
        results["recall_scores"].append(recall)
        results["f1_scores"].append(f1)
        
        # Store category performance
        category = test_case['category']
        if category not in results["category_performance"]:
            results["category_performance"][category] = []
        results["category_performance"][category].append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        
        # Detailed results
        detailed_result = {
            "query": test_case['query'],
            "expected": test_case['expected_diseases'],
            "predicted": predicted_diseases,
            "full_response": generated_text,
            "metrics": {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        }
        results["detailed_results"].append(detailed_result)
        
        print(f"سوال: {test_case['query']}")
        print(f"پاسخ مدل: {generated_text}")
        print(f"بیماری‌های پیش‌بینی شده: {predicted_diseases}")
        print(f"بیماری‌های مورد انتظار: {test_case['expected_diseases']}")
        print(f"دقت: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print("-" * 80)

    # Calculate overall metrics
    overall_metrics = {
        "mean_accuracy": np.mean(results["accuracy_scores"]),
        "mean_precision": np.mean(results["precision_scores"]),
        "mean_recall": np.mean(results["recall_scores"]),
        "mean_f1": np.mean(results["f1_scores"]),
        "std_accuracy": np.std(results["accuracy_scores"]),
        "std_precision": np.std(results["precision_scores"]),
        "std_recall": np.std(results["recall_scores"]),
        "std_f1": np.std(results["f1_scores"])
    }

    # Category-wise performance
    for category, performances in results["category_performance"].items():
        category_metrics = {
            "accuracy": np.mean([p["accuracy"] for p in performances]),
            "precision": np.mean([p["precision"] for p in performances]),
            "recall": np.mean([p["recall"] for p in performances]),
            "f1": np.mean([p["f1"] for p in performances])
        }
        results["category_performance"][category] = category_metrics

    results["overall_metrics"] = overall_metrics

    # Print comprehensive results
    print("\n" + "="*50)
    print("📊 نتایج ارزیابی جامع")
    print("="*50)
    print(f"دقت کلی: {overall_metrics['mean_accuracy']:.3f} ± {overall_metrics['std_accuracy']:.3f}")
    print(f"Precision کلی: {overall_metrics['mean_precision']:.3f} ± {overall_metrics['std_precision']:.3f}")
    print(f"Recall کلی: {overall_metrics['mean_recall']:.3f} ± {overall_metrics['std_recall']:.3f}")
    print(f"F1-Score کلی: {overall_metrics['mean_f1']:.3f} ± {overall_metrics['std_f1']:.3f}")
    
    print("\n📈 عملکرد بر اساس دسته‌بندی:")
    for category, metrics in results["category_performance"].items():
        print(f"{category}: Acc={metrics['accuracy']:.3f}, P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

    # Save results
    with open(f"{Config.FINAL_MODEL_DIR}/evaluation_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def calculate_disease_accuracy(predicted: List[str], expected: List[str]) -> float:
    """Calculate accuracy for disease prediction"""
    if not expected:
        return 1.0 if not predicted else 0.0
    
    # Normalize disease names for comparison
    predicted_normalized = [disease.strip().lower() for disease in predicted]
    expected_normalized = [disease.strip().lower() for disease in expected]
    
    # Calculate intersection
    correct_predictions = len(set(predicted_normalized) & set(expected_normalized))
    total_expected = len(expected_normalized)
    
    return correct_predictions / total_expected if total_expected > 0 else 0.0

def calculate_disease_metrics(predicted: List[str], expected: List[str]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score for disease prediction"""
    if not expected and not predicted:
        return 1.0, 1.0, 1.0
    if not expected:
        return 0.0, 1.0, 0.0
    if not predicted:
        return 1.0, 0.0, 0.0
    
    # Normalize disease names
    predicted_normalized = set(disease.strip().lower() for disease in predicted)
    expected_normalized = set(disease.strip().lower() for disease in expected)
    
    # Calculate metrics
    true_positives = len(predicted_normalized & expected_normalized)
    false_positives = len(predicted_normalized - expected_normalized)
    false_negatives = len(expected_normalized - predicted_normalized)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def interactive_test(model_path: str):
    """Interactive testing interface"""
    
    print("بارگذاری مدل برای تست تعاملی...")
    
    config = PeftConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        load_in_4bit=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("\n🩺 دستیار پزشکی آماده است! (برای خروج 'exit' تایپ کنید)")
    
    while True:
        user_input = input("\nعلائم خود را شرح دهید: ").strip()
        
        if user_input.lower() in ['exit', 'خروج', 'quit']:
            break
            
        if not user_input:
            continue
        
        prompt = f"""<|system|>
شما یک دستیار پزشکی متخصص هستید که بر اساس علائم ارائه شده، بیماری‌های محتمل را تشخیص می‌دهید.

<|user|>
{user_input}

<|assistant|>
با توجه به علائم ذکر شده، بیماری‌های محتمل عبارتند از:"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.4,
                do_sample=True,
                top_p=0.85,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response.split("<|assistant|>")[-1].strip()
        
        print(f"\n🩺 پاسخ دستیار پزشکی:\n{generated_text}")

def main():
    """Main execution function"""
    
    # Set environment variables for better performance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("🚀 شروع آموزش مدل پزشکی با بهبودهای جامع")
    
    # Load model and tokenizer
    model, tokenizer = load_base_model()
    
    # Prepare data
    dataset_path = "/kaggle/input/multi-intend-detection/multi_intent_disease_queries(method3).json"
    tokenized_dataset = prepare_training_data_enhanced(dataset_path, tokenizer)
    print(f"📊 آماده‌سازی داده‌ها: {tokenized_dataset}")
    
    # Create LoRA config
    lora_config = create_optimized_lora_config()
    
    # Train model
    trained_model, history = train_model_enhanced(model, tokenizer, tokenized_dataset, lora_config)
    
    print("✅ آموزش کامل شد!")
    
    # Comprehensive evaluation
    print("\n🔍 شروع ارزیابی جامع...")
    evaluation_results = evaluate_model_comprehensive(Config.FINAL_MODEL_DIR)
    
    # Interactive testing
    print("\n🎯 آغاز تست تعاملی...")
    interactive_test(Config.FINAL_MODEL_DIR)
    
    return trained_model, history, evaluation_results

if __name__ == "__main__":
    # Run the improved training and evaluation
    trained_model, training_history, eval_results = main()