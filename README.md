# 👩‍💻 Fine-tune Phi-3-mini model to generate Python Code

## phi3-mini-python-code-20k

**Phi3-mini** model fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** using the method **LoRA** with [PEFT](https://github.com/huggingface/peft) and bitsandbytes library. We also save the adapter.

Link to the model in Huggingface: https://huggingface.co/edumunozsala/phi3-mini-python-code-20k

Link to the adapter in Huggingface: https://huggingface.co/edumunozsala/phi-3-mini-LoRA

## phi3-mini-4k-qlora-python-code-20k

**Phi3-mini** model fine-tuned on the **python_code_instructions_18k_alpaca Code instructions dataset** using **QLoRA** in 4-bit with [PEFT](https://github.com/huggingface/peft) and bitsandbytes library.

Link to the model in Huggingface:: https://huggingface.co/edumunozsala/phi3-mini-4k-qlora-python-code-20k

Link to the adapter in Huggingface: https://huggingface.co/edumunozsala/phi-3-mini-QLoRA

## The dataset

For our tuning process, we will take a [dataset](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca) containing about 18,000 examples where the model is asked to build a Python code that solves a given task. 
This is an extraction of the [original dataset](https://huggingface.co/datasets/sahil2801/code_instructions_120k) where only the Python language examples are selected. Each row contains the description of the task to be solved, an example of data input to the task if applicable, and the generated code fragment that solves the task is provided.

## Problem description

Our goal is to fine-tune the pretrained model, Phi3-mini 3.8B parameters, using both the PEFT method, **LoRA**, and a 4-bit quantization **QLoRA**to produce a Python coder.Then we'll evluate the performance of both models. We will run the training on Google Colab using a A100 to get better performance. But you can try out to run it on a T4 adjusting some parameters to reduce memory consumption like batch size.

## The base models

[Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

The Phi-3-Mini-4K-Instruct is a 3.8B parameters, lightweight, state-of-the-art open model trained with the Phi-3 datasets that includes both synthetic data and the filtered publicly available websites data with a focus on high-quality and reasoning dense properties. The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support.

The model has underwent a post-training process that incorporates both supervised fine-tuning and direct preference optimization for the instruction following and safety measures. When assessed against benchmarks testing common sense, language understanding, math, code, long context and logical reasoning, Phi-3 Mini-4K-Instruct showcased a robust and state-of-the-art performance among models with less than 13 billion parameters.


## Content

- Fine-tuning notebook using LoRA: `phi-3-finetune-lora-python-coder.ipynb`
- Fine-tuning notebook using QLoRA: `phi-3-finetune-qlora-python-coder.ipynb`

### Example of usage

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "edumunozsala/phi3-mini-4k-qlora-python-code-20k"
device_map="cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", device_map=device_map)

input="'Create a function to calculate the sum of a sequence of integers.\n Input: [1, 2, 3, 4, 5]'"

# Create the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
# Prepare the prompt or input to the model
prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": input}], tokenize=False, add_generation_prompt=True)
# Run the pipe to get the answer
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95,
                   max_time= 180)
print(outputs[0]['generated_text'][len(prompt):].strip())

```
### Citation

```
@misc {edumunozsala_2024,
	author       = { {Eduardo Muñoz} },
	title        = { phi3-mini-4k-qlora-python-code-20k },
	year         = 2023,
	url          = { https://huggingface.co/edumunozsala/phi3-mini-4k-qlora-python-code-20k },
	publisher    = { Hugging Face }
}
```
## Contributing
If you find some bug or typo, please let me know or fixit and push it to be analyzed. 

## License

Copyright 2023 Eduardo Muñoz

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
