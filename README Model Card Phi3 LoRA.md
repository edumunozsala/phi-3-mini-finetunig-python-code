---
tags:
- generated_from_trainer
- code
- coding
- llama-2
model-index:
- name: phi3-mini-python-code-20k
  results: []
license: apache-2.0
language:
- code
datasets:
- iamtarun/python_code_instructions_18k_alpaca
pipeline_tag: text-generation
---


# Phi-3-mini 3.8B LoRA Python Coder 👩‍💻 

**Phi-3-mini 3.8B** fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** by using the method **LoRA** with [PEFT](https://github.com/huggingface/peft) library.

## Pretrained description

[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support.

## Tokenizer
Phi-3 Mini-4K-Instruct supports a vocabulary size of up to 32064 tokens. The tokenizer files already provide placeholder tokens that can be used for downstream fine-tuning, but they can also be extended up to the model's vocabulary size.

## Training data

[python_code_instructions_18k_alpaca](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca)

The dataset contains problem descriptions and code in python language. This dataset is taken from sahil2801/code_instructions_120k, which adds a prompt column in alpaca style.

### Chat Format
Given the nature of the training data, the Phi-3 Mini-4K-Instruct model is best suited for prompts using the chat format as follows. You can provide the prompt as a question with a generic template as follow:

```
<|user|>\nQuestion <|end|>\n<|assistant|>
```

For example:

```
<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>
```

where the model generates the text after <|assistant|> . In case of few-shots prompt, the prompt can be formatted as the following:
```
<|user|>
I am going to Paris, what should I see?<|end|>
<|assistant|>
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."<|end|>
<|user|>
What is so great about #1?<|end|>
<|assistant|>
```

### Training hyperparameters

The following `PEFT`  config was used during training:
```py
################################################################################
# LoRA parameters
################################################################################
# LoRA attention dimension
lora_r = 16
# Alpha parameter for LoRA scaling
lora_alpha = 16
# Dropout probability for LoRA layers
lora_dropout = 0.05
# Modules
target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
```

**SFTTrainer arguments**
```py
        evaluation_strategy="steps",
        do_eval=True,
        optim="adamw_torch",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=100,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        report_to="wandb",
```

### Framework versions
- PEFT 0.4.0

## Evaluation
Evaluating on a test dataset of 500 samples:

```text
Rouge 1 Mean:  56.65322508234244
Rouge 2 Mean:  37.547274096577084
Rouge L Mean:  51.08407579855678
Rouge Lsum Mean:  56.256016384803075

```

### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/phi3-mini-python-code-20k"

tokenizer = AutoTokenizer.from_pretrained(hf_model_repo,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(hf_model_repo, trust_remote_code=True, torch_dtype="auto", device_map="cuda")


instruction="Create an algorithm in Python to sort an array of numbers."
input="[9, 3, 5, 1, 6]"

prompt = f"""### Instruction:
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Output:
"""

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to execute inference on a prompt
def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
                   max_time= 180) #, eos_token_id=eos_token)
    return outputs[0]['generated_text'][len(prompt):].strip()


test_inference(prompt)

```

### Citation

```
@misc {edumunozsala_2023,
	author       = { {Eduardo Muñoz} },
	title        = { phi3-mini-python-code-20k },
	year         = 2024,
	url          = { https://huggingface.co/edumunozsala/phi3-mini-python-code-20k },
	publisher    = { Hugging Face }
}
```