# Key Changes:
# Adaptive Stopping: Added uncertainty_threshold parameter and early exit condition
# Score Normalization: Implemented normalization using standard deviation per Appendix C
# Log Prob Normalization: Added per-token normalization in SourceModel.get_log_prob
# Maintained Functionality: Preserved tqdm progress bars and core functionality
# Configurability: Made uncertainty threshold configurable while keeping default values practical


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer

from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from bert_score import score as bert_score
from typing import List
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")



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
        
        return [text] + [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]





class BertScoreKernel(Kernel):
    def __init__(self, sim_matrix: torch.Tensor):
        super().__init__()
        self.sim_matrix = sim_matrix
        self.register_parameter(
            name="raw_alpha", 
            parameter = nn.Parameter(torch.tensor(0.0)))
        self.register_parameter(
            name="raw_beta", 
            parameter = nn.Parameter(torch.tensor(0.0)))
        
    @property
    def alpha(self):
        return F.softplus(self.raw_alpha)

    @property
    def beta(self):
        return F.softplus(self.raw_beta)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
        x1 = x1.long().squeeze()
        x2 = x2.long().squeeze()
        
        x1_idx = x1.unsqueeze(-1)
        x2_idx = x2.unsqueeze(0)
        
        K = self.alpha * self.sim_matrix[x1_idx, x2_idx] + self.beta
        return K





class SurrogateModel(ExactGP):
    def __init__(self, sim_matrix: torch.Tensor, train_indices: torch.Tensor, train_y: torch.Tensor):
        likelihood = GaussianLikelihood()
        super().__init__(train_indices, train_y, likelihood)
        self.mean_module = ZeroMean()    #  Can also try ConstantMean()
        self.covar_module = BertScoreKernel(sim_matrix)
        
    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        x = x.long()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def optimize_hyperparameters(self, train_x, train_y, lr=0.01, iterations=50):   #   May tune this "iteration" to 100
        self.train()
        self.likelihood.train()
        
        optimizer = Adam([
            {'params': self.covar_module.parameters()},
            {'params': self.likelihood.parameters()}
        ], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        for _ in range(iterations):
            optimizer.zero_grad()
            output = self(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()





class BayesianDetector:
    def __init__(self, source_model, perturbation_model="t5-3b", num_total_samples=10, uncertainty_threshold=0.1):
        self.num_total_samples = num_total_samples
        self.source_model = source_model
        self.perturbator = PerturbationGenerator(perturbation_model)
        self.gp = None
        self.all_texts = []
        self.uncertainty_threshold = uncertainty_threshold

    def get_log_prob(self, texts: List[str]) -> torch.Tensor:
        return self.source_model.get_log_prob(texts)
    
    def detect(self, text: str, max_queries: int = 5) -> float:
        self.all_texts = self.perturbator.perturb(
            text, 
            num_samples=self.num_total_samples-1
        )
        max_queries = min(max_queries, self.num_total_samples - 2)
        
        num_texts = len(self.all_texts)
        sim_matrix = torch.zeros((num_texts, num_texts))
        with tqdm(total=((num_texts+1)*(num_texts) // 2), desc="Precomputing BERTScore") as pbar:
            for i in range(num_texts):
                for j in range(i, num_texts):
                    if i == j:
                        sim_matrix[i, j] = 1.0
                    else:
                        _, _, F1 = bert_score(
                            [self.all_texts[i]], [self.all_texts[j]],
                            lang="en",
                            model_type="bert-base-uncased",
                            rescale_with_baseline=True,
                            verbose=False
                        )
                        sim_matrix[i, j] = sim_matrix[j, i] = F1.item()
                    pbar.update(1)
        
        train_indices = torch.tensor([0, 1])
        y_train = self.get_log_prob([self.all_texts[0], self.all_texts[1]])
        
        for _ in tqdm(range(max_queries - 2), desc="Active Sampling"):
            self.gp = SurrogateModel(sim_matrix, train_indices, y_train)
            self.gp.optimize_hyperparameters(train_indices, y_train)
            
            remaining_indices = torch.tensor(
                [i for i in range(num_texts) if i not in train_indices]
            )
            
            with torch.no_grad():
                self.gp.eval()
                pred = self.gp.likelihood(self.gp(remaining_indices))
                uncertainties = pred.variance
                

            #   Here we are selecting the pertubations with the highest uncertainity
            if torch.max(uncertainties) < self.uncertainty_threshold:
                break
                
            next_idx = remaining_indices[torch.argmax(uncertainties)].item()
            train_indices = torch.cat([train_indices, torch.tensor([next_idx])])
            y_train = torch.cat([y_train, self.get_log_prob([self.all_texts[next_idx]])])
        
        with torch.no_grad():
            self.gp.eval()
            all_preds = self.gp(torch.arange(num_texts))
            
        original_score = all_preds.mean[0]
        perturbed_scores = all_preds.mean[1:]
        perturbed_mean = perturbed_scores.mean()
        perturbed_std = perturbed_scores.std()
        
        if perturbed_std == 0:
            perturbed_std = 1e-8
        
        normalized_score = (original_score - perturbed_mean) / perturbed_std
        return normalized_score.item()



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
            
        print(log_probs)    
        return torch.tensor(log_probs)



if __name__ == "__main__":
    source_model = SourceModel(model_name="gpt2")  
    
    detector = BayesianDetector(
        source_model,
        perturbation_model="t5-3b",
        num_total_samples=10,
        uncertainty_threshold=1e-3
    )
    
    sentence = "The rapid development of artificial intelligence has brought both opportunities and challenges to modern society."
    detection_score = detector.detect(sentence, max_queries=5)
    print(f"Detection score: {detection_score}")


    #   num_samples -> Affects the no. of perturbations (Refer PerturbationGenerator)
    #   max_queries -> Affects the no. of times active sampling is done (Refer detect method of BayesianDetector)
    #   uncertainity_threshold -> Tune it to select perturbations accordingly

    #   Classification Threshold: If Detection Score > 1.4 => LLM generated, else Human-written