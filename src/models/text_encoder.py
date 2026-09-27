import hashlib
import re
import torch
import torch.nn as nn
from .attention import TransformerBlock


class LightweightClinicalTextEncoder(nn.Module):
    """
    Offline/lightweight clinical text encoder based on deterministic SHA-256 vocabulary hashing
    and a multi-head self-attention transformer.
    Guarantees deterministic, cross-process execution without requiring external model weights.
    """
    def __init__(self, vocab_size=8192, embed_dim=128, depth=2, num_heads=4, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=2.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        torch.nn.init.normal_(self.pos_embed, std=0.02)

    def _tokenize(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        token_ids = []
        for w in words[:self.max_len]:
            # Stable deterministic SHA-256 hash to vocab space (independent of PYTHONHASHSEED)
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest(), 16) % (self.vocab_size - 1) + 1  # 0 reserved for pad
            token_ids.append(h)
        if len(token_ids) == 0:
            token_ids = [1]
        # Pad or truncate
        if len(token_ids) < self.max_len:
            token_ids = token_ids + [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return token_ids

    def forward_text(self, texts, device):
        token_matrix = [self._tokenize(t) for t in texts]
        tokens = torch.tensor(token_matrix, dtype=torch.long, device=device)
        return self.forward(tokens)

    def forward(self, tokens):
        # tokens: (B, max_len)
        B, L = tokens.shape
        x = self.token_embed(tokens) + self.pos_embed[:, :L, :]
        mask = (tokens != 0).unsqueeze(-1)  # (B, L, 1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # Masked average pooling
        x_masked = x * mask
        sum_pooled = x_masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        text_emb = sum_pooled / denom
        return text_emb


class ClinicalReportEncoder(nn.Module):
    """
    Clinical Text Encoder with support for HuggingFace pretrained models
    (e.g. sentence-transformers/all-MiniLM-L6-v2 or Bio_ClinicalBERT) with automatic fallback
    to LightweightClinicalTextEncoder for guaranteed offline reproducibility.
    
    Scientific Note:
    - Default sentence-transformers/all-MiniLM-L6-v2 is a general-domain semantic sentence transformer,
      chosen for efficient CPU embedding.
    - Specialized clinical models (e.g. emilyalsentzer/Bio_ClinicalBERT) can be supplied via model_name.
    """
    def __init__(self, model_name=None, embed_dim=128, device=None):
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.hf_model = None
        self.tokenizer = None
        self.is_hf = False

        if model_name:
            try:
                from transformers import AutoTokenizer, AutoModel
                print(f"Loading HuggingFace text encoder: {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.hf_model = AutoModel.from_pretrained(model_name)
                # Freeze backbone weights to preserve pretrained semantic representations
                for p in self.hf_model.parameters():
                    p.requires_grad = False
                self.hf_model.eval()
                self.hf_dim = self.hf_model.config.hidden_size
                self.hf_proj = nn.Linear(self.hf_dim, embed_dim)
                self.is_hf = True
                print("HuggingFace text encoder loaded and frozen successfully.")
            except Exception as e:
                print(f"Warning: Could not load {model_name} ({e}). Falling back to LightweightClinicalTextEncoder.")
                self.is_hf = False

        if not self.is_hf:
            self.fallback_encoder = LightweightClinicalTextEncoder(embed_dim=embed_dim)

    @property
    def active_encoder_name(self):
        if self.is_hf:
            return f"HuggingFace: {self.model_name} (general-domain semantic sentence transformer)"
        return "LightweightClinicalTextEncoder (deterministic offline fallback)"

    def forward(self, reports):
        """
        reports: list of str, or tensor of token IDs
        returns: (B, embed_dim)
        """
        if isinstance(reports, str):
            reports = [reports]

        if self.is_hf and isinstance(reports, list):
            inputs = self.tokenizer(
                reports, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.hf_model(**inputs)
            # Mean pooling with attention mask
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)
            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            denom = attention_mask.sum(dim=1).clamp(min=1e-6)
            pooled = sum_embeddings / denom
            return self.hf_proj(pooled)
        elif isinstance(reports, list):
            device = next(self.parameters()).device
            return self.fallback_encoder.forward_text(reports, device)
        else:
            return self.fallback_encoder(reports)
