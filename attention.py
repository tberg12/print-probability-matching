import math
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, num_input_filters, dropout, args):
        super(Attention, self).__init__()
        self.num_input_filters = num_input_filters
        # self.attn_dim = attn_dim  # w * h
        self.dropout = dropout
        self.args = args

        if self.args.per_channel_attention:
            self.mlp = nn.Sequential(
                nn.Linear(self.num_input_filters, self.num_input_filters),
                nn.Tanh(),
                nn.Linear(self.num_input_filters, self.num_input_filters),
                nn.Tanh(),
                nn.Linear(self.num_input_filters, self.num_input_filters),
                nn.Tanh(),
                nn.Linear(self.num_input_filters, self.num_input_filters // 2)
            )
        else:
            self.mlp = nn.Sequential(
                # 64, 16*16, 2*64, 1, 1
                nn.Linear(self.num_input_filters, self.num_input_filters),
                nn.Tanh(),
                nn.Linear(self.num_input_filters, self.num_input_filters // 2),
                nn.Tanh(),
                nn.Linear(self.num_input_filters // 2, self.num_input_filters // 4),
                nn.Tanh(),
                nn.Linear(self.num_input_filters // 4, 1)
            )

    def forward(self, embed1, embed2, transform_embeddings=False):
        """
        :param embed1: convolutional feature patches for img1
        :param embed2: convolutional feature patches for img2
        :param transform_embeddings: whether to apply a linear layer to embed1/embed2 prior to doing MLP attention
        :param just_max_over_alpha: max directly over the alphas (in height x width dim) to get an output score 
                (as opposed to returning pointwise (cross-weighted) embeddings for each img)
        """
        if transform_embeddings:
            transformed_embed1 = self.input_transform(embed1)  # (batch_size, attn_dim)
            transformed_embed2 = self.input_transform(embed2)  # (batch_size, attn_dim)
        else:
            transformed_embed2 = embed2
            transformed_embed1 = embed1
        batch_size = embed1.shape[0]
        num_filters = embed1.shape[1]
        h, w = embed1.shape[2], embed1.shape[3]
        # fold height and width dims into batch dim to enable h*w forward passes in a single forward pass
        # embed1: (batch_size, num_filters, h, w)
        embed1_collapsed = embed1.permute(0, 2, 3, 1).contiguous().view(-1, num_filters)
        # embed2_collapsed = embed2.view(batch_size * h * w, num_filters)
        embed2_collapsed = embed2.permute(0, 2, 3, 1).contiguous().view(-1, num_filters)
        if self.args.per_channel_attention:
            scores = self.mlp(torch.cat([embed1_collapsed, embed2_collapsed], dim=1))  # (batch_size * h * w, num_filters)
            scores = scores.view(batch_size, h * w, num_filters).permute(0, 2, 1).contiguous()  # (batch_size, num_filters, h * w)
            weights = torch.softmax(scores / self.args.softmax_temperature, dim=2)  # (batch_size, num_filters, h*w)
        else:
            # pass 2*num_filters vector to MLP to compare concatenated CNN output feature patches across single i,j location (batch_size * h * w, 2*num_filters)
            scores = self.mlp(torch.cat([embed1_collapsed, embed2_collapsed], dim=1))  # (batch_size * h * w, 1)
            weights = torch.softmax(scores.view(batch_size, h * w) / self.args.softmax_temperature, dim=1)  # (batch_size, h*w)
        # if self.args.just_max_over_alphas:
            # TODO: do we have a proper receptive field size here for direct maxing to get score?
            # max_pooled_alphas = torch.max(scores.view(batch_size, h * w), dim=-1)[0]
            # return weights.squeeze(), max_pooled_alphas
        # else: # pointwise multiply weights
        if self.args.per_channel_attention:
            context_embed1 = weights * embed1.view(batch_size, num_filters, -1) # (batch_size, num_filters, h * w)
            context_embed2 = weights * embed2.view(batch_size, num_filters, -1)
        else:
            # unsqueeze weights in num_filters dim to allow broadcasting
            context_embed1 = weights.unsqueeze(1) * embed1.view(batch_size, num_filters, -1) # (batch_size, num_filters, h * w)
            context_embed2 = weights.unsqueeze(1) * embed2.view(batch_size, num_filters, -1)
        
        if self.args.use_output_embedding_layer:  # just output the weighted feature maps so we can pass them through an output layer later
            return weights.squeeze(), context_embed1, context_embed2  # _, (batch_size, num_filters, h * w)

        # context_embed{1,2} dims: (batch_size, num_filters, h * w)
        # if self.args.combine_attn_embeddings_with_euclidean_diff:
        #     if self.args.collapse_attn_operation == 'filter_sum':
        #         embed = (context_embed1 - context_embed2).pow(2).sum(1)
        #     elif self.args.collapse_attn_operation == 'filter_max':
        #         embed = (context_embed1 - context_embed2).pow(2).max(1)[0]
        #     elif self.args.collapse_attn_operation == 'hw_sum':
        #         embed = (context_embed1 - context_embed2).pow(2).sum(-1)
        #     elif self.args.collapse_attn_operation == 'hw_max':
        #         embed = (context_embed1 - context_embed2).pow(2).max(-1)[0]
        #     return weights.squeeze(), embed

        if self.args.collapse_attn_operation == 'filter_sum':
            context_embed1 = context_embed1.sum(1)
            context_embed2 = context_embed2.sum(1)
        elif self.args.collapse_attn_operation == 'filter_max':
            context_embed1 = context_embed1.max(1)[0]
            context_embed2 = context_embed2.max(1)[0]
        elif self.args.collapse_attn_operation == 'hw_sum':
            context_embed1 = context_embed1.sum(-1)
            context_embed2 = context_embed2.sum(-1)
        elif self.args.collapse_attn_operation == 'hw_max':
            context_embed1 = context_embed1.max(-1)[0]
            context_embed2 = context_embed2.max(-1)[0]

        # sum over h*w to get 64 embed
        # if self.args.combine_attn_embeddings_with_sum:
            # return weights.squeeze(), context_embed1 + context_embed2
        # else:
        return weights.squeeze(), context_embed1, context_embed2
