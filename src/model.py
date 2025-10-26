import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class CodeLLM:
    def __init__(self, repo_id: str = "codellama/CodeLlama-7b-Instruct-hf"):
        # 4‑bit quantisation – massive memory saving
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            device_map="auto",
            quantization_config=quant_cfg,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        # Compile for faster inference on recent PyTorch
        if torch.cuda.is_available():
            self.model = torch.compile(self.model, mode="max-autotune")

    def complete(self, prompt: str, max_new_tokens: int = 256,
                 temperature: float = 0.2, stop: list[str] | None = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            stop_token_ids=[self.tokenizer.convert_tokens_to_ids(s) for s in (stop or [])],
        )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        # Strip the original prompt
        return text[len(prompt):].lstrip()