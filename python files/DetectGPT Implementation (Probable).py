import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import List


class PerturbationGenerator:
    def __init__(self, model_name: str = "t5-3b"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = 512
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def perturb(self, text: str, num_samples: int = 9) -> List[str]:
        inputs = self.tokenizer(
            f"paraphrase: {text}", return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_return_sequences=num_samples,
            do_sample=True,
            max_length=512,
        )
        
        return [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]



class SourceModel:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        
    def get_log_prob(self, texts: List[str]) -> torch.Tensor:
        log_probs = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
                
                # Handle empty input
                if inputs["input_ids"].shape[1] == 0:
                    log_probs.append(0.0)
                    continue
                
                # Handle single-token edge case
                num_tokens = inputs["input_ids"].shape[1]
                if num_tokens == 1:
                    log_probs.append(0.0)
                    continue
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                logits = outputs.logits
                labels = inputs["input_ids"]
                
                log_probs_seq = torch.log_softmax(logits, dim=-1)[:, :-1, :]
                log_probs_text = torch.gather(
                    log_probs_seq, 
                    dim=2, 
                    index=labels[:, 1:, None]
                ).squeeze(-1).sum(dim=1)

                normalized_log_prob = log_probs_text.item() / (num_tokens - 1)  
                log_probs.append(normalized_log_prob)
            
        # print(log_probs)    
        return torch.tensor(log_probs)
    


class ScoreCalculator:
    def __init__(self):
        self.perturb_model = PerturbationGenerator(model_name='t5-3b')
        self.source_model = SourceModel(model_name='gpt-2')

    def score(self, candidate_text):
        perturbations = self.perturb_model.perturb(text=candidate_text, num_samples=200)     #   Increase for better accuracy
        candidate_text_log_prob = self.source_model.get_log_prob([candidate_text])
        perturbations_log_prob = self.source_model.get_log_prob(perturbations)

        detection_score = candidate_text_log_prob - perturbations_log_prob.mean()
        return detection_score



if __name__ == '__main__':
    sc = ScoreCalculator()
    sentence = "Hello, this is Saranya Pal...!!"
    print(sc.score(candidate_text=sentence))