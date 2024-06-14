import time
from abc import abstractmethod
import os
from openai import OpenAI
from peft import PeftModel
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer



class LLM:
    @abstractmethod
    def generate(self, messages, **model_config) -> str:
        pass

class LLM_OpenAI(LLM):
    def __init__(self, model='gpt-3.5-turbo-0301'):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model

    def generate(self, messages, **model_config):
        ret = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            **model_config
        )
        return ret.choices[0].message.content

class LLM_guanaco_33b(LLM):
    def __init__(self, model_name='huggyllama/llama-30b'):
        self.model_name = model_name

    def generate(self, messages, **model_config):
        adapters_name = "timdettmers/guanaco-33b"
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # offload_folder="/home/ec2-user/SageMaker/hf_cache",
            max_memory={i: "16384MB" for i in range(torch.cuda.device_count())},
        )
        model = PeftModel.from_pretrained(model, adapters_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        max_new_tokens = 200

        prompt = ""
        for msg in messages:
            if msg['role'] == "user":
                prompt = msg['content']
        formatted_prompt = (
            f"A chat between a curious human and an artificial intelligence assistant."
            f"The assistant gives helpful, concise, and polite answers to the user's questions.\n"
            f"### Human: {prompt} ### Assistant:"
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda:0")
        outputs = model.generate(inputs=inputs.input_ids, max_new_tokens=max_new_tokens)
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        res_sp = res.split("###")
        output = res_sp[1] + res_sp[2]

        return output

class LLM_falcon_7b(LLM):
    def __init__(self, model_name='tiiuae/falcon-7b-instruct'):
        self.model_name = model_name

    def generate(self, messages, **model_config):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        max_new_tokens = 200
        prompt = ""
        for msg in messages:
            if msg['role'] == "user":
                prompt = msg['content']
        sequences = pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            res = seq["generated_text"]

        return res
