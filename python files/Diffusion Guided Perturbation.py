import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)

from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

from bert_score import score as bert_score
from typing import List, Optional
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")



# ================== NEW DIFFUSION COMPONENTS ==================
class DiffusionRefiner:
    def __init__(self, model_name: str = "microsoft/trocr-small-stage1"):
        """Text refinement using diffusion-inspired process"""
        self.refiner = pipeline(
            "text2text-generation", 
            model=model_name, 
            device=0 if torch.cuda.is_available() else -1
        )
        self.paraphraser = pipeline(
            "text2text-generation",
            model="humarin/chatgpt_paraphraser_on_T5_base",
            device=0 if torch.cuda.is_available() else -1
        )

    def _diffusion_step(self, text: str, strength: float = 0.3) -> str:
        """Simulate diffusion process through iterative denoising"""
        # Step 1: Add noise through paraphrasing
        noised = self.paraphraser(
            text,
            do_sample=True,
            top_k=50,
            num_return_sequences=1,
            max_length=512
        )[0]['generated_text']
        
        # Step 2: Refine using reconstruction model
        refined = self.refiner(
            noised,
            max_length=512,
            num_beams=3,
            early_stopping=True
        )[0]['generated_text']
        
        return refined if np.random.rand() < strength else text

    def refine(self, texts: List[str], steps: int = 2) -> List[str]:
        """Apply diffusion-inspired refinement to text batch"""
        return [self._diffusion_step(t, strength=0.3) for t in texts]
    


# ================== MODIFIED PERTURBATION GENERATOR ==================
class PerturbationGenerator:
    def __init__(self, model_name: str = "t5-base", use_diffusion: bool = True):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.diffusion = DiffusionRefiner() if use_diffusion else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def perturb(self, text: str, num_samples: int = 9) -> List[str]:
        """Generate perturbations with optional diffusion refinement"""
        # Generate initial perturbations with T5
        inputs = self.tokenizer(
            f"paraphrase: {text}", 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            num_return_sequences=num_samples,
            do_sample=True,
            max_length=512,
        )
        
        raw_perturbations = [text] + [
            self.tokenizer.decode(out, skip_special_tokens=True) 
            for out in outputs
        ]
        
        # Apply diffusion refinement if enabled
        if self.diffusion:
            try:
                refined = self.diffusion.refine(raw_perturbations[1:])
                return [raw_perturbations[0]] + refined
            except Exception as e:
                print(f"Diffusion refinement failed: {e}, using raw perturbations")
        
        return raw_perturbations
    



# ================== CORE DETECTION FRAMEWORK (MODIFIED) ==================
class BayesianDetector:
    def __init__(
        self,
        source_model,
        perturbation_model: str = "t5-base",
        num_total_samples: int = 10,
        uncertainty_threshold: float = 0.1,
        use_diffusion: bool = True
    ):
        self.num_total_samples = num_total_samples
        self.source_model = source_model
        self.perturbator = PerturbationGenerator(
            perturbation_model, 
            use_diffusion=use_diffusion
        )
        self.gp = None
        self.all_texts = []
        self.uncertainty_threshold = uncertainty_threshold
        self.sim_matrix_cache = {}

    # ================== HELPER METHODS ==================
    def _compute_similarity_matrix(self, texts: List[str]) -> torch.Tensor:
        """Precompute BERTScore similarity matrix with caching"""
        cache_key = hash(tuple(texts))
        if cache_key in self.sim_matrix_cache:
            return self.sim_matrix_cache[cache_key]
        
        n = len(texts)
        sim_matrix = torch.zeros((n, n))
        for i in tqdm(range(n), desc="Computing Similarities"):
            for j in range(i, n):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    _, _, F1 = bert_score(
                        [texts[i]], [texts[j]],
                        lang="en",
                        model_type="bert-base-uncased",
                        rescale_with_baseline=True,
                        verbose=False
                    )
                    sim_matrix[i, j] = sim_matrix[j, i] = F1.item()
        
        self.sim_matrix_cache[cache_key] = sim_matrix
        return sim_matrix


    # ================== CORE DETECTION LOGIC ==================
    def detect(self, text: str, max_queries: int = 5) -> float:
        """Enhanced detection with diffusion-guided perturbations"""
        self.all_texts = self.perturbator.perturb(
            text, 
            num_samples=self.num_total_samples-1
        )
        max_queries = min(max_queries, self.num_total_samples - 2)
        
        # Precompute similarity matrix
        sim_matrix = self._compute_similarity_matrix(self.all_texts)
        
        # Active sampling loop
        train_indices = torch.tensor([0, 1])
        y_train = self.source_model.get_log_prob([self.all_texts[0], self.all_texts[1]])
        
        for _ in tqdm(range(max_queries - 2), desc="Active Sampling"):
            self.gp = SurrogateModel(sim_matrix, train_indices, y_train)
            self.gp.optimize_hyperparameters(train_indices, y_train)
            
            remaining_indices = torch.tensor(
                [i for i in range(len(self.all_texts)) if i not in train_indices]
            )
            
            with torch.no_grad():
                self.gp.eval()
                pred = self.gp.likelihood(self.gp(remaining_indices))
                uncertainties = pred.variance
                
            # Early stopping condition
            if torch.max(uncertainties) < self.uncertainty_threshold:
                break
                
            next_idx = remaining_indices[torch.argmax(uncertainties)].item()
            train_indices = torch.cat([train_indices, torch.tensor([next_idx])])
            y_train = torch.cat([y_train, self.source_model.get_log_prob([self.all_texts[next_idx]])])
        
        # Final prediction
        with torch.no_grad():
            self.gp.eval()
            all_preds = self.gp(torch.arange(len(self.all_texts)))
            
        original_score = all_preds.mean[0]
        perturbed_scores = all_preds.mean[1:]
        perturbed_std = perturbed_scores.std()
        
        # Stability guard
        if perturbed_std < 1e-8:
            perturbed_std = 1e-8
            
        return (original_score - perturbed_scores.mean()) / perturbed_std
    



# ================== REMAINING ORIGINAL COMPONENTS ==================
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
    # Example usage
    source_model = SourceModel(model_name="gpt2")  
    
    # With diffusion refinement
    detector_diff = BayesianDetector(
        source_model,
        perturbation_model="t5-base",
        use_diffusion=True,
        num_total_samples=10,
        uncertainty_threshold=1e-3
    )
    
    # Without diffusion refinement
    detector_raw = BayesianDetector(
        source_model,
        use_diffusion=False,
        num_total_samples=10,
        uncertainty_threshold=1e-3
    )
    
    test_text = "The rapid development of artificial intelligence has brought both opportunities and challenges to modern society."
    
    print("Running detection with diffusion refinement...")
    score_diff = detector_diff.detect(test_text)
    print(f"Diffusion-enhanced score: {score_diff:.2f}")
    
    print("\nRunning detection without diffusion refinement...")
    score_raw = detector_raw.detect(test_text)
    print(f"Raw T5 score: {score_raw:.2f}")