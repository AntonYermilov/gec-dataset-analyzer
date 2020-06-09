from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel


class PerplexityScorer(ABC):
    @abstractmethod
    def tokenize_sent(self, sent: str):
        pass

    @abstractmethod
    def evaluate_ppl(self, tokenize_ids) -> float:
        pass

    @abstractmethod
    def evaluate_loss(self, tokenize_ids) -> float:
        pass


class OpenAIGPTLMHeadModelPpl(OpenAIGPTLMHeadModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def forward(self, input_ids, position_ids=None, token_type_ids=None, lm_labels=None):
        assert lm_labels is not None

        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        lm_logits = self.lm_head(hidden_states)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()

        # Flatten the tokens
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_labels.view(-1))
        return loss


class GPT2PerplexityScorer(PerplexityScorer):
    def __init__(self):
        self.model = OpenAIGPTLMHeadModelPpl.from_pretrained('openai-gpt')
        self.model.eval()
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def tokenize_sent(self, sent: str):
        tokenize_input = self.tokenizer.tokenize(sent)
        tokenize_ids = self.tokenizer.convert_tokens_to_ids(tokenize_input)
        return tokenize_ids

    @torch.no_grad()
    def evaluate_ppl(self, tokenize_ids) -> float:
        tensor_input = torch.tensor([tokenize_ids])
        loss = self.model(tensor_input, lm_labels=tensor_input)
        return np.exp(loss.item() / len(tensor_input))

    @torch.no_grad()
    def evaluate_loss(self, tokenize_ids) -> float:
        tensor_input = torch.tensor([tokenize_ids])
        loss = self.model(tensor_input, lm_labels=tensor_input)
        return loss.item()
