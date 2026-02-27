# Gemma 3 Arabic Empathy Fine-tuning

This project fine-tunes `unsloth/gemma-3-4b-it` using LoRA to produce a more empathetic Arabic conversational model.

## 🔹 Model
Base: Unsloth/gemma-3-4b-it  
Method: LoRA (Unsloth)  
Precision: 4-bit training → merged → GGUF export  

## 🔹 Dataset
Arabic empathetic conversations dataset.

## 🔹 Results
Improved emotional alignment and supportive tone in Arabic responses.

## 🔹 Model
[https://huggingface.co/Lokatsu/gemma3-arabic-empathy](https://huggingface.co/Lokatsu/gemma-3-Arabic-Empathy)

## 🔹 GGUF Model
Available here:
https://huggingface.co/Lokatsu/gemma3-arabic-empathy-gguf

## 🔹 Training
```bash
python gemma3_finetune.py
