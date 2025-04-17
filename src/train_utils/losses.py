import torch.nn.functional as F

def soft_cross_entropy(logits, target):
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()