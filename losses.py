import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NPairsWithExtraNegatives(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, anchor, positive, extra_negatives=None):
        # Normalize all vectors
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        # Similarities between anchor and positives: [B, B]
        pos_logits = anchor @ positive.T

        # Remove diagonal from positives for in-batch negatives
        B = anchor.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=anchor.device)
        in_batch_neg_logits = pos_logits[mask].view(B, -1)

        logits = in_batch_neg_logits  # [B, B-1]

        # If extra negatives exist, compute anchor-to-extra-neg sim
        if extra_negatives is not None:
            extra_negatives = F.normalize(extra_negatives, p=2, dim=1)
            extra_logits = anchor @ extra_negatives.T  # [B, K]
            logits = torch.cat([in_batch_neg_logits, extra_logits], dim=1)  # [B, B-1+K]

        # Positive logits (correct match): diagonal of anchor @ positive.T
        pos_logit = torch.sum(anchor * positive, dim=1, keepdim=True)  # [B, 1]

        # Final logits: [B, 1+num_negatives]
        logits = torch.cat([pos_logit, logits], dim=1)

        # Targets: correct positive is always at index 0
        targets = torch.zeros(B, dtype=torch.long, device=anchor.device)

        # Cross entropy over [pos | negs]
        return F.cross_entropy(logits, targets)
    


class ContrastiveLoss(nn.Module):
    def __init__(self, initial_temperature=0.07, learnable_temperature=False):
        """
        Contrastive loss with optional extra negatives.
        """
        super(ContrastiveLoss, self).__init__()
        assert not learnable_temperature
        self.temperature = initial_temperature
    
    def get_temperature(self):
        return torch.tensor(self.temperature)

    def forward(self, anchor_features, positive_features, extra_negative_features=None):
        """
        Args:
            anchor_features (Tensor): shape (batch_size, dim)
            positive_features (Tensor): shape (batch_size, dim)
            extra_negative_features (Tensor, optional): shape (num_extra_negatives, dim)
                Additional negatives not paired with any anchors.

        Returns:
            loss (Tensor): scalar loss
        """
        # Normalize all features
        anchor_features = F.normalize(anchor_features, dim=1)
        positive_features = F.normalize(positive_features, dim=1)

        batch_size = anchor_features.size(0)

        if extra_negative_features is not None:
            extra_negative_features = F.normalize(extra_negative_features, dim=1)

            # Combine positives and extra negatives as potential matches
            all_pos_features = torch.cat([positive_features, extra_negative_features], dim=0)  # shape: (batch_size + num_extra_negatives, dim)
        else:
            all_pos_features = positive_features

        # Compute similarity logits
        logits = anchor_features @ all_pos_features.T  # shape: (batch_size, batch_size + num_extra_negatives)

        # Apply temperature
        logits /= self.temperature

        # Targets point to correct positives only (first `batch_size` columns)
        targets = torch.arange(batch_size, device=anchor_features.device)

        # Compute loss only for correct matches in the first `batch_size` columns
        loss_a2p = F.cross_entropy(logits, targets)

        # Optional symmetric loss (positive to anchor)
        if extra_negative_features is not None:
            all_anchor_features = torch.cat([anchor_features, extra_negative_features], dim=0)
        else:
            all_anchor_features = anchor_features

        logits_t = positive_features @ all_anchor_features.T  # shape: (batch_size, batch_size + num_extra_negatives)
        logits_t /= self.temperature
        loss_p2a = F.cross_entropy(logits_t, targets)

        return (loss_a2p + loss_p2a) / 2
    

# class ContrastiveLoss(nn.Module):
#     def __init__(self, initial_temperature=0.07, learnable_temperature=False):
#         """
#         Contrastive loss with optional extra negatives and a learnable temperature.
#         """
#         super(ContrastiveLoss, self).__init__()
#         if learnable_temperature:
#             self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / initial_temperature)))
#         else:
#             self.register_buffer('log_temp', torch.log(torch.tensor(1.0 / initial_temperature)))
#         self.learnable_temperature = learnable_temperature

#     def get_temperature(self):
#         return torch.exp(self.log_temp)

#     def forward(self, anchor_features, positive_features, extra_negative_features=None):
#         """
#         Args:
#             anchor_features (Tensor): (batch_size, dim)
#             positive_features (Tensor): (batch_size, dim)
#             extra_negative_features (Tensor, optional): (num_extra_negatives, dim)

#         Returns:
#             loss (Tensor): scalar contrastive loss
#         """
#         anchor_features = F.normalize(anchor_features, dim=1)
#         positive_features = F.normalize(positive_features, dim=1)

#         batch_size = anchor_features.size(0)

#         if extra_negative_features is not None:
#             extra_negative_features = F.normalize(extra_negative_features, dim=1)
#             all_pos_features = torch.cat([positive_features, extra_negative_features], dim=0)
#         else:
#             all_pos_features = positive_features

#         logits = anchor_features @ all_pos_features.T
#         temperature = self.get_temperature()
#         logits /= temperature

#         targets = torch.arange(batch_size, device=anchor_features.device)
#         loss_a2p = F.cross_entropy(logits, targets)

#         if extra_negative_features is not None:
#             all_anchor_features = torch.cat([anchor_features, extra_negative_features], dim=0)
#         else:
#             all_anchor_features = anchor_features

#         logits_t = positive_features @ all_anchor_features.T
#         logits_t /= temperature
#         loss_p2a = F.cross_entropy(logits_t, targets)

#         return (loss_a2p + loss_p2a) / 2


class CrossEncoderTripletBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, sim_ap, sim_an):
        """
        Args:
            sim_ap (torch.Tensor): Model output similarity logits for anchor-positive pairs (batch_size,).
            sim_an (torch.Tensor): Model output similarity logits for anchor-negative pairs (batch_size,).
        """
        label_ap = torch.ones_like(sim_ap)  # Label 1 for anchor-positive
        label_an = torch.zeros_like(sim_an)  # Label 0 for anchor-negative
        loss_ap = self.bce_loss(sim_ap, label_ap)
        loss_an = self.bce_loss(sim_an, label_an)
        total_loss = loss_ap + loss_an
        return total_loss


class CrossEncoderTripletMarginLoss(nn.Module):
    def __init__(self, margin, reduction="mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, sim_ap, sim_an):
        """
        Args:
            sim_ap (torch.Tensor): Similarity logits for anchor-positive pairs (batch_size,).
            sim_an (torch.Tensor): Similarity logits for anchor-negative pairs (batch_size,).
        """
        # We want the positive similarity to be higher than the negative similarity by at least margin.
        # Thus, we penalize if (sim_ap - sim_an) < margin.
        loss = F.relu(self.margin - (sim_ap - sim_an))
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("reduction must be one of 'mean', 'sum', or 'none'")
        return loss


class TripletScoreLoss(nn.Module):
    """
    Triplet loss with scores (not embeddings)
    Takes distance values for positive and negative pairs
    """

    def __init__(self, margin, reduction):
        super(TripletScoreLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, distance_positive, distance_negative):
        losses = F.relu(distance_positive - distance_negative + self.margin)
        assert len(losses) == len(distance_positive) == len(distance_negative)
        
        if self.reduction=="mean":
            result = losses.mean()
        elif self.reduction=="sum":
            result = losses.sum()
        elif self.reduction=="none":
            result = losses
        else:
            ValueError("`reduction` should be one of {none, mean, sum}")
        return result


class TripletDualEncoderLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, reduction, squared=True):
        super(TripletDualEncoderLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.squared = squared

    def forward(self, anchor, positive, negative):
        if self.squared:
            distance_positive = (anchor - positive).pow(2).sum(1) 
            distance_negative = (anchor - negative).pow(2).sum(1) 
        else:
            distance_positive = (anchor - positive).sum(1) 
            distance_negative = (anchor - negative).sum(1) 
        losses = F.relu(distance_positive - distance_negative + self.margin)
        assert len(losses) == len(anchor)
        
        if self.reduction == "mean":
            result = losses.mean()
        elif self.reduction == "sum":
            result = losses.sum()
        elif self.reduction == "none":
            result = losses
        else:
            ValueError("`reduction` should be one of {none, mean, sum}")
        return result


class TripletDualEncoderLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, reduction, squared=True):
        super(TripletDualEncoderLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.squared = squared

    def forward(self, pos_anchor, positive, neg_anchor, negative):
        if self.squared:
            distance_positive = (pos_anchor - positive).pow(2).sum(1) 
            distance_negative = (neg_anchor - negative).pow(2).sum(1) 
        else:
            distance_positive = (pos_anchor - positive).sum(1) 
            distance_negative = (neg_anchor - negative).sum(1) 
        losses = F.relu(distance_positive - distance_negative + self.margin)
        assert len(losses) == len(pos_anchor)
        assert len(losses) == len(neg_anchor)
        
        if self.reduction == "mean":
            result = losses.mean()
        elif self.reduction == "sum":
            result = losses.sum()
        elif self.reduction == "none":
            result = losses
        else:
            ValueError("`reduction` should be one of {none, mean, sum}")
        return result
    