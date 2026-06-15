"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
From: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, eq1_loss=False):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)  # (B, B)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)  # (B, B)

        contrast_count = features.shape[1]  # V
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B, V, F) -> (B*V, F)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # (B, F)
            anchor_count = 1
        elif self.contrast_mode == 'all':  # default behavior
            anchor_feature = contrast_feature  # (B*V, F)
            anchor_count = contrast_count  # V
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits  (for numerator/denominator dot product term)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),  # 'all': (B*V, F) x (F, B*V) = (B*V, B*V),   'one'=(B, F) x (F, B*V) = (B, B*V)
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask (in V dim)
        mask = mask.repeat(anchor_count, contrast_count)  # 'all': (B*V, B*V),     'one': (B, B*V)
        # mask-out self-contrast cases
        all_logits_mask = torch.scatter(
            torch.ones_like(mask),  # (B*V, B*V)
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),  # (B*V, 1) indices
            0
        )
        # import ipdb; ipdb.set_trace()
        positive_mask = mask * all_logits_mask
        numerator_logits = logits
        if eq1_loss:
            numerator_logits = positive_mask * logits

        exp_logits = torch.exp(logits) * all_logits_mask  # denominator
        log_prob = numerator_logits - torch.log(exp_logits.sum(1, keepdim=True))  # numerator - denominator

        # compute mean of log-likelihood over positive  aka   1/|P(i)| * \Sum_{p \in P(i)}
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss