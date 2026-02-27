import unsloth
from unsloth import FastLanguageModel
import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer



# ----------------------------
# Config
# ----------------------------
MODEL_NAME = "unsloth/gemma-3-4b-it"   # Unsloth-optimized Gemma 3 4B IT 
DATASET_NAME = "arbml/arabic_empathetic_conversations" 

OUTPUT_DIR = "gemma3-4b-it-arabic-empathy-lora"

MAX_SEQ_LENGTH = 1024   
LOAD_IN_4BIT = False     # QLoRA-style
SEED = 42

# ----------------------------
# 1) Load model + tokenizer
# ----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = torch.bfloat16, 
    load_in_4bit = LOAD_IN_4BIT,
)

# LoRA setup
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    lora_alpha = 64,
    lora_dropout = 0,
    bias = "none",
    target_modules = [
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    use_gradient_checkpointing = "unsloth",
    random_state = SEED,
)

# ----------------------------
# 2) Load dataset
# ----------------------------
ds = load_dataset(DATASET_NAME)

# Dataset has a "train" split in the viewer (36.6k rows).
# We'll create a validation split.
ds = ds["train"].train_test_split(test_size=0.02, seed=SEED)
train_ds, eval_ds = ds["train"], ds["test"]

# ----------------------------
# 3) Prompt engineering (Arabic empathic assistant)
# ----------------------------
SYSTEM_PROMPT = (
    "أنت مساعد عربي متعاطف وداعم. "
    "استمع جيدًا، اعترف بمشاعر المستخدم، وقدم ردًا لطيفًا وعمليًا إن أمكن. "
    "تجنّب إطلاق الأحكام. اجعل الرد طبيعيًا وبالعربية."
)

def format_example(ex):
    """
    Map each row -> one chat sample for Gemma SFT:
      system + user(context [+ label]) -> assistant(response)
    """
    context = (ex.get("context") or "").strip()
    response = (ex.get("response") or "").strip()
    label = ex.get("label", None)

    include_label = True
    if include_label and label is not None:
        user_content = f"{context}\n\n(وسم المشاعر: {label})"
    else:
        user_content = context

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]

    # Apply the model's chat template.
    # We set add_generation_prompt=False because assistant text is included (SFT).
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(format_example,  remove_columns=eval_ds.column_names)

# ----------------------------
# 4) Training arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,   
    learning_rate=2e-4,
    warmup_ratio=0.03,
    num_train_epochs=2,              
    logging_steps=20,
    eval_strategy="steps",
    eval_steps=800,
    save_steps=800,
    save_total_limit=3,
    bf16=True,                       
    optim="adamw_8bit",              
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=SEED,
    report_to="none",
)

# ----------------------------
# 5) Trainer
# ----------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=True,
    args=training_args,
)

trainer.train()

# ----------------------------
# 6) Save LoRA adapter
# ----------------------------
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nSaved LoRA adapter to: {OUTPUT_DIR}")

# ----------------------------
# 7) Quick inference test (after training)
# ----------------------------
FastLanguageModel.for_inference(model)

test_messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "أشعر بضيق شديد لأنني فقدت وظيفتي اليوم."},
]

prompt = tokenizer.apply_chat_template(
    test_messages, tokenize=False, add_generation_prompt=True
)

inputs = tokenizer(text=[prompt], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

print("\n--- Model reply ---")
print(tokenizer.decode(out[0], skip_special_tokens=True))