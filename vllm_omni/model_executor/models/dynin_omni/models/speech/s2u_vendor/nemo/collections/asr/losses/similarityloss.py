import torch

from nemo.core import Loss


class NegativeCosineSimilarityLoss(Loss):

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction == 'mean'
        self.reduction = reduction

    def forward(self, predictions: torch.tensor, targets: torch.tensor):
        similarity_scores = torch.cosine_similarity(predictions.float(), targets.float(), dim=-1).type_as(predictions)
        loss = 1.0 - similarity_scores.mean()
        return loss
