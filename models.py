import matching_losses
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import math
import torchvision
from supconloss import SupConLoss
import wandb
from collections import OrderedDict
import attention
from torch.distributions.categorical import Categorical
from torchvision import models
import timm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StackedNet(nn.Module):
    """
    Baseline of stacking two images in 2 channels and
    classifying as damage match or not
    """
    def __init__(self, args):
        super(StackedNet,self).__init__()
        act = nn.ReLU()
        self.args = args
        make_conv_block = lambda ic, oc, k, s, p: nn.Sequential(
            nn.Conv2d(ic, oc, k, s, p),
            nn.BatchNorm2d(oc),
            act,
        )
        make_initial_conv_mp_block = lambda c, ck, s, p, mpk, mps, mpp: nn.Sequential(
            make_conv_block(2, c//2, ck, s, p),
            make_conv_block(c//2, c, ck, s, p),
            nn.MaxPool2d((mpk,mpk), (mps,mps),  (mpp,mpp)),
        )
        make_internal_conv_mp_block = lambda c, ck, s, p, mpk, mps, mpp, n: nn.Sequential(
            *(make_conv_block(c, c, ck, s, p),)*n,
            nn.MaxPool2d((mpk,mpk), (mps,mps), (mpp,mpp)),
            )
        self.cnn_dim = args.cnn_dim
        h = self.cnn_dim
        self.output_dim = None
        if 'cnn_output_dim' in args:
            self.output_dim = args.cnn_output_dim  # damage_classifier
        elif 'embedding_size' in args:
            self.output_dim = args.embedding_size  # matcher
        # self.cnn = nn.Sequential(
        #     make_initial_conv_mp_block(h, 3, 1, 1, 2, 2, 0),
        #     make_internal_conv_mp_block(h, 3, 1, 1, 3, 3, 0, 1),
        #     make_conv_block(h, h, 3, 1, 1),
        #     make_internal_conv_mp_block(h, 2, 1, 1, 4, 4, 0, 1),
        #     nn.Flatten(),
        #     nn.Linear(512, 1),
        #     nn.Sigmoid()
        # )
        modules = [
            make_initial_conv_mp_block(h, 3, 1, 1, 2, 2, 0)
        ] + ([
            make_internal_conv_mp_block(h, 3, 1, 1, 2, 2, 0, self.args.num_conv_per_block)
        ] * (self.args.num_cnn_blocks - 1)) + \
        [
            nn.Flatten(),
            nn.Linear(self.cnn_dim * 16, 1),
            nn.Sigmoid()
        ]
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)
    
    def pair_distance(self, x):
        return self.forward(x)


class DamageCNN(nn.Module):
    """Our standard CNN for 64 x 64 aligned b/w character images
    used for both damage detection and matching
    """
    def __init__(self, args, num_channels=1, use_linear_head=False):
        super(DamageCNN, self).__init__()
        self.args = args
        self.num_channels = num_channels
        self.mlp_activation = nn.ReLU()
        act = nn.ReLU()
        make_conv_block = lambda ic, oc, k, s, p: nn.Sequential(
            nn.Conv2d(ic, oc, k, s, p),
            nn.BatchNorm2d(oc),
            act,
        )
        make_initial_conv_mp_block = lambda c, ck, s, p, mpk, mps, mpp: nn.Sequential(
            make_conv_block(1, c//2, ck, s, p),
            make_conv_block(c//2, c, ck, s, p),
            nn.MaxPool2d(mpk, mps, mpp),
        )
        make_internal_conv_mp_block = lambda c, ck, s, p, mpk, mps, mpp, n: nn.Sequential(
            *(make_conv_block(c, c, ck, s, p),)*n,
            nn.MaxPool2d(mpk, mps, mpp),
        )
        self.cnn_dim = args.cnn_dim
        h = self.cnn_dim
        self.output_dim = None
        if 'cnn_output_dim' in args:
            self.output_dim = args.cnn_output_dim  # damage_classifier
        elif 'embedding_size' in args:
            self.output_dim = args.embedding_size  # matcher
        self.use_linear_head = use_linear_head
        modules = [
            make_initial_conv_mp_block(h, 3, 1, 1, 2, 2, 0)
        ] + [
            make_internal_conv_mp_block(h, 3, 1, 1, 2, 2, 0, self.args.num_conv_per_block)
        ] * (self.args.num_cnn_blocks - 1)
        self.cnn = nn.Sequential(*modules)
        
        self.flatten = nn.Flatten()
        if self.use_linear_head:
            self.head = nn.Sequential(
                nn.Linear(self.cnn_dim * 4 * 4, self.output_dim),
                nn.BatchNorm1d(self.output_dim)
            )

    def forward(self, x):
        out_dict = {}
        out_dict['conv'] = self.cnn(x)
        out_dict['conv_flat'] = nn.Flatten()(out_dict['conv'])
        if self.use_linear_head:
            out_dict['proj_output'] = self.head(out_dict['conv_flat'])
        return out_dict



class TripletDualEncoderNet(nn.Module):
    """ A rather standard dual encoder network for matching
    NOTE: most hyperparams/options are contained in the args param
    """
    def __init__(self, args, num_channels=1):
        super(TripletDualEncoderNet, self).__init__()
        self.args = args
        self.num_channels = num_channels
        self.mlp_dropout = nn.Dropout(p=args.mlp_dropout)
        self.avg_image = None
        self.avg_image_batch_jittered = torch.zeros((args.batch_size, 1, 64, 64)).to(device)
        self.cnn = DamageCNN(args, use_linear_head=False)
        self.head = nn.Sequential(
                nn.Linear(self.cnn.cnn_dim * 4 * 4, self.cnn.output_dim),
                nn.BatchNorm1d(self.cnn.output_dim)
            )

    def forward(self, x):
        if self.args.input_residual and self.avg_image is not None:
            if self.training and self.args.jitter_triplet:
                x = (x - self.avg_image_batch_jittered.to(device)[:x.shape[0]])  #.pow(2)
            else:
                x = (x - self.avg_image.expand_as(x).to(device))  #.pow(2)
        out_dict = self.cnn(x)
        out = out_dict['conv_flat']
        if self.args.conv_template_residual:
            out = out_dict['conv']
            if self.training and self.args.jitter_triplet:
                avg_feat = self.cnn(self.avg_image_batch_jittered.to(device)[:out.shape[0]])['conv']
            else:
                avg_feat = self.cnn(self.avg_image.to(device))['conv']
            out = (out - avg_feat)  # .pow(2)
            out = nn.Flatten()(out)
        out_dict['proj_output'] = self.head(out)
        return out_dict['proj_output']
     
    def get_embedding(self, x):
        return self.forward(x)

    def pair_distance(self, pair):  # for evaluation
        anchor_embedding = self.forward(pair[:, :1])
        other_embedding = self.forward(pair[:, 1:])
        return (anchor_embedding - other_embedding).pow(2).sum(1)


def expand_to_3_channels(x: torch.tensor):
    """ if x is one channel, repeat it to 3 channels """
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x

class DualEncoderNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.encoder_type == 'vit_b_16':
            self.encoder = models.vit_b_16(pretrained=args.use_pretrained)
        elif 'hf_hub' in args.encoder_type:
            self.encoder = timm.create_model(args.encoder_type, num_classes=0, pretrained=args.use_pretrained)
        else:
            raise ValueError(f'Invalid encoder type for DualEncoderNet: {args.encoder_type}')

    def forward(self, x):
        x = expand_to_3_channels(x)  # [B, 1, 224, 224] -> [B, 3, 224, 224]
        out = self.encoder(x)  # ViT: [B, 768]  ConvNext: [B, 1024] 
        return out
     
    def get_embedding(self, x):
        return self.forward(x)

    def pair_distance(self, pair):  # for evaluation
        anchor_embedding = self.forward(pair[:, :1])
        other_embedding = self.forward(pair[:, 1:])
        return (anchor_embedding - other_embedding).pow(2).sum(1)


class CrossEncoderNet(nn.Module):
    def __init__(self, 
                 args,
                 embed_dim=768,  # default for all ViT models
                 pooling_method='max'):  # 'mean', 'max', or 'cls'
        super().__init__()
        if args.encoder_type == 'vit_b_16':
            self.encoder = models.vit_b_16(pretrained=args.use_pretrained)
        elif 'hf_hub' in args.encoder_type and 'convnext' not in args.encoder_type:
            self.encoder = timm.create_model(args.encoder_type, num_classes=0, pretrained=args.use_pretrained)
        else:
            raise ValueError(f'Invalid encoder type for CrossEncoderNet: {args.encoder_type}')
        self.encoder.head = nn.Identity()  # Remove the classification head
        # Pooling method for combining token embeddings
        self.pooling_method = pooling_method
        # Task-specific head for computing similarity
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def pool_features(self, tokens):
        """
        Pool token embeddings based on the selected pooling method.
        """
        if self.pooling_method == 'mean':
            return tokens.mean(dim=1)  # Mean pooling
        elif self.pooling_method == 'max':
            return tokens.max(dim=1).values  # Max pooling
        elif self.pooling_method == 'cls':
            return tokens[:, 0]  # Use CLS token
        else:
            raise ValueError("Invalid pooling method. Choose from 'mean', 'max', or 'cls'.")

    def forward(self, img1, img2):
        img1 = expand_to_3_channels(img1)
        img2 = expand_to_3_channels(img2)
        # Pass each image through the patch embedding and transformer
        tokens1 = self.encoder.forward_features(img1)  # Shape: [B, N, D], where N=785 for patch8, N=197 for patch16
        tokens2 = self.encoder.forward_features(img2)  # Shape: [B, N, D], where N=785 for patch8, N=197 for patch16
        # Concatenate token embeddings from both images
        combined_tokens = torch.cat((tokens1, tokens2), dim=1)  # Shape: [B, 2N, D]
        # Use the ViT's transformer encoder to compute cross-attention
        cross_attention_tokens = self.encoder.blocks(combined_tokens)  # Shape: [B, 2N, D]
        # Pool the features
        pooled_features = self.pool_features(cross_attention_tokens)  # Shape: [B, D]
        # Compute similarity using the task-specific head
        logits = self.fc(pooled_features)  # Shape: [B, 1]
        if self.training:
            return logits
        else:
            return torch.sigmoid(logits)


class AttentionNet(nn.Module):
    """ An attention based dual encoder for matching
    """
    def __init__(self, args, num_channels=1, convnet=None):
        """ NOTE: most hyperparams/options are contained in the args param
        """
        super(AttentionNet, self).__init__()
        self.args = args
        self.num_channels = num_channels
        self.mlp_dropout = nn.Dropout(p=args.mlp_dropout)
        self.avg_image = None
        self.avg_image_batch_jittered = torch.zeros((args.batch_size, 1, 64, 64)).to(device)
        self.net = DamageCNN(args)

        if self.args.use_output_embedding_layer:
            self.output_embedding_layer = nn.Sequential(nn.Flatten(), nn.Linear(in_features=16 * 16 * 64, out_features=self.embedding_size, bias=True))

        # TODO: dropout in attention
        # attention operates on the two feature maps of each image so we double the output feature width of the 
        self.attention_net = attention.Attention(2 * self.net.output_dim, self.mlp_dropout, self.args)

    def forward(self, img1, img2):
        if self.args.input_residual and self.avg_image is not None:
            if self.training and self.args.jitter_triplet:
                img1 = (img1 - self.avg_image_batch_jittered.to(device)[:img1.shape[0]])  #.pow(2)
                img2 = (img2 - self.avg_image_batch_jittered.to(device)[:img2.shape[0]])  #.pow(2)
            else:
                img1 = (img1 - self.avg_image.expand_as(img1).to(device))  #.pow(2)
                img2 = (img2 - self.avg_image.expand_as(img2).to(device))  #.pow(2)
        feat1 = self.net(img1)['conv']
        feat2 = self.net(img2)['conv']
        if self.args.conv_template_residual:
            if self.training and self.args.jitter_triplet:
                avg_feat = self.net(self.avg_image_batch_jittered.to(device)[:feat1.shape[0]])['conv']
            else:
                avg_feat = self.net(self.avg_image.expand_as(img1).to(device))['conv']
            feat1 = (feat1 - avg_feat)  #.pow(2)
            feat2 = (feat2 - avg_feat)  #.pow(2)
        attn_out = self.compute_attention(feat1, feat2)
        attn_out, weights = attn_out[:-1], attn_out[-1]
        if self.args.per_channel_attention:
            # TODO: fix this; just takes first filter weights for now to get it to work
            attn_entropy = Categorical(probs=weights[:, 0, :]).entropy().mean().item()
        else:
            attn_entropy = Categorical(probs=weights).entropy().mean().item()
        if self.args.use_output_embedding_layer:
            attn_out = (self.output_embedding_layer(attn_out[0]), 
                        self.output_embedding_layer(attn_out[1]))
        # assert not self.args.combine_attn_embeddings_with_sum
        # attn_feat1, attn_feat2 = attn_out
        out = attn_out
        return out, attn_entropy, weights
    
    def compute_attention(self, embed1, embed2):
        out = self.attention_net(embed1, embed2)
        weights, out = out[0], out[1:]
        if len(out) == 1 and isinstance(out, tuple):
            out = out[0]
        # if self.args.just_max_over_alphas:
        #     return out, weights  # just a score (float)
        # elif self.args.combine_attn_embeddings_with_sum:
        #     return out, weights
        # elif self.args.combine_attn_embeddings_with_euclidean_diff:
        #     return out, weights
        # else:
        context_embed1, context_embed2 = out
        return context_embed1, context_embed2, weights
    
    def get_embedding(self, x):
        return self.forward(x)

    def pair_distance(self, pair):  # for evaluation (don't return attn_entropy fro self.forward)
        anchor_embedding, other_embedding = self.forward(pair[:, :1], pair[:, 1:])[0]
        return (anchor_embedding - other_embedding).pow(2).sum(1)
        

#
#  damage detection networks below
#

class RPN(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1,), subnet_kernel_size: int = 3):
        super().__init__()
        #
        # from:
        # https://github.com/facebookresearch/detectron2/blob/a406e69f6687b9e3923db31bfda89c339e0a81c4/detectron2/modeling/proposal_generator/rpn.py#L67
        # args: (in_channels: int, num_anchors: int = 1, box_dim: int = 4, conv_dims: List[int] = (-1,))
        # 
        cur_channels = in_channels
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                self.conv.add_module(f"relu{k}", nn.ReLU())
                cur_channels = out_channels
        # 1x1 conv for predicting objectness logits
        self.object_classification_subnet = nn.Conv2d(cur_channels, num_anchors, kernel_size=subnet_kernel_size, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_offset_regression_subnet = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=subnet_kernel_size, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    
    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features: list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas
    

class ProjectionNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, net_type='linear'):
        super().__init__()
        self.net_type = net_type
        if net_type == 'mlp' and hidden_dim is None:
            hidden_dim = out_dim
        self.net = nn.Sequential()
        if net_type == 'linear':
            self.net.add_module('output', nn.Linear(in_dim, out_dim))
        elif net_type == 'mlp':
            self.net.add_module('hidden', nn.Linear(in_dim, hidden_dim))
            self.net.add_module('relu', nn.ReLU())
            self.net.add_module('output', nn.Linear(hidden_dim, out_dim))
            
    def forward(self, x):
        out = self.net(x)
        return F.normalize(out, dim=-1)


class Subnet(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, box_dim: int = 4, conv_dims: List[int] = (-1, -1, -1, -1), subnet_kernel_size: int = 3, final_bias_init_pi: float = None, add_box_regression_layer: bool = False):
        super().__init__()
        cur_channels = in_channels
        self.subnet_kernel_size = subnet_kernel_size
        self.box_dim = box_dim
        self.num_anchors = num_anchors  # A
        self.conv_dims = conv_dims
        self.conv = nn.Sequential()
        for k, conv_dim in enumerate(conv_dims):
            out_channels = cur_channels if conv_dim == -1 else conv_dim
            if out_channels <= 0:
                raise ValueError(
                    f"Conv output channels should be greater than 0. Got {out_channels}"
                )
            conv = self._get_subnet_conv(cur_channels, out_channels)
            self.conv.add_module(f"conv{k}", conv)
            self.conv.add_module(f"relu{k}", nn.ReLU())
            cur_channels = out_channels
        
        if add_box_regression_layer:
            self.conv.add_module("flatten", nn.Flatten())
            self.conv.add_module("box_regression_linear", nn.Linear(cur_channels, self.box_dim * self.num_anchors, bias=True))

        # All new conv layers except the final one in the RetinaNet subnets are 
        # initialized with bias b = 0 and a Gaussian weight fill with σ = 0.01.
        conv_layer_num = 0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                conv_layer_num += 1
                nn.init.normal_(layer.weight, std=0.01)
                # For the final conv layer of the classification subnet, 
                # we set the bias initialization to b = − log((1 − π)/π), 
                # where π specifies that at the start of training every anchor 
                # should be labeled as foreground with confidence of ∼π. 
                # We use π = .01 in all experiments, although results 
                # are robust to the exact value.
                if final_bias_init_pi is not None and conv_layer_num == len(conv_dims):
                    nn.init.constant_(layer.bias, -np.log((1 - final_bias_init_pi) / final_bias_init_pi))
                else:
                    nn.init.constant_(layer.bias, 0)

    def _get_subnet_conv(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
class CBAMResNet(nn.Module):
    def __init__(self, args):
        super(CBAMResNet, self).__init__()
        self.args = args
        import cbam_model
        self.cnn = cbam_model.ResidualNet('CIFAR10', 18, 1, 'CBAM')

    def forward(self, x, **kwargs):
        cnn_feats = self.cnn(torchvision.transforms.Resize((32, 32))(x['image']).repeat((1, 3, 1, 1)).to(device))
        out_dict = {'cnn_feats': cnn_feats}
        if self.args.use_rpn:
            # rpn
            region_props = self.rpn(cnn_feats)
            out_dict['region_props'] = region_props
        elif self.args.use_attn:
            # NOTE: just do max over filters now
            attn_input = cnn_feats.max(1)[0].transpose(0, 1)
            out_dict['attention_logits'], out_dict['attention_weights'] = self.multihead_attn(attn_input, attn_input, attn_input)
            out_dict['attention_logits'] = out_dict['attention_logits'].transpose(0, 1)
        #out_dict['logits'] = self.fc_out(self.dropout(self.flatten(cnn_feats)))
        # do global average pooling like Squeezenet (pool over h/w for each channel)
        if self.args.output_pooling == 'global_avg_pool_hw':
            out_dict['logits'] = self.fc_out(torch.mean(cnn_feats.view(cnn_feats.size(0), cnn_feats.size(1), -1), dim=2))
        elif self.args.output_pooling == 'max_pool_f':
            raise NotImplementedError
            # out_dict['logits'] = self.classify(torch.max(cnn_feats, dim=1))  # TODO
        elif self.args.output_pooling == 'none' and self.args.loss_type in {'object_ce', 'focal'}:
            if self.args.use_template_residual:
                template_cnn_feats = self.cnn(kwargs['template_image'][:1].to(device) if 'template_image' in kwargs else x['avg_template'][:1].to(device))
                out_dict['template_cnn_feats'] = template_cnn_feats
                cnn_feats -= template_cnn_feats
            out_dict['logits'] = self.classify(cnn_feats)
        else:
            # out_dict['logits'] = self.fc_out(self.dropout(self.flatten(cnn_feats)))
            out_dict['logits'] = cnn_feats  # torch.exp(cnn_feats)
        # out_dict['cam'] = self.compute_cam(cnn_feats.detach().cpu().numpy(), torch.sigmoid(out_dict['logits']).detach().cpu().numpy())
        return out_dict

    def compute_loss(self, output, target, reduction='mean', **kwargs):
        #loss = model.compute_loss(out_dict, label, damage_loc_xy=damage_loc_xy)
        logits = output['logits']
        if self.args.loss_type == 'ce':
            ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target.to(logits.device).type_as(logits), reduction=reduction)
            return ce_loss
        elif self.args.loss_type == "object_ce":
            xy_targets = []
            xy_logits = []
            # TODO: get one negative box grid off char skeleton and use in loss to make it balanced!
            xys = [
                (x.item(), y.item()) 
                for x, y in zip(*kwargs['damage_loc_xy'])
            ]
            non_dam_xys = [
                (x.item(), y.item()) 
                for x, y in zip(*kwargs['non_damage_loc_xy'])
            ]
            # import ipdb; ipdb.set_trace()
            for b in range(len(xys)):
                x, y = xys[b]
                if x == -1 or y == -1:
                    # NOTE: skip negative examples!
                    continue
                    xy_logits.append(logits[b])
                    xy_targets.append(torch.zeros((1, 1, 16, 16), dtype=torch.float, device=device).view(-1))
                else:
                    xy_logits.append(logits[b])
                    # import ipdb; ipdb.set_trace()
                    xy_targets_pos = self.make_target_grid(xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1)
                    xy_targets_neg = self.make_target_grid(non_dam_xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1)
                    # xy_targets.append((xy_targets_pos, xy_targets_neg))
                    xy_targets.append(xy_targets_pos)
                    
            xy_targets = torch.stack(xy_targets)
            xy_logits = torch.stack(xy_logits)
            # NOTE: only compute loss on positive examples or what?
            # import ipdb; ipdb.set_trace()
            # pos_weight = torch.full((1,), (1-xy_targets).sum() / xy_targets.sum()).to(device)
            pos_weight = self.args.pos_weight_ce_multiplier * torch.full((1,), (1-xy_targets).sum() / xy_targets.sum()).to(device) if self.args.use_pos_weight_on_ce and xy_targets.sum() > 0.0 else None
            # ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), xy_targets.to(logits.device), reduction=reduction, pos_weight=pos_weight)
            ce_loss = F.binary_cross_entropy_with_logits(xy_logits.squeeze(), xy_targets.to(logits.device), reduction=reduction, pos_weight=pos_weight)
            # import ipdb; ipdb.set_trace()
            return ce_loss
        elif self.args.loss_type == 'focal':
            xy_targets = []
            xys = [(x.item(), y.item()) for x, y in zip(*kwargs['damage_loc_xy'])]
            for b in range(target.shape[0]):
                x, y = xys[b]
                if x == -1 or y == -1:
                    xy_targets.append(torch.zeros((1, 1, 16, 16), dtype=torch.float, device=device).view(-1))
                else:
                    xy_targets.append(self.make_target_grid(xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1))
            # import ipdb; ipdb.set_trace()
            xy_targets = torch.stack(xy_targets)
            return matching_losses.sigmoid_focal_loss(logits.squeeze(), xy_targets.to(logits.device).type_as(logits), reduction=reduction)
        elif self.args.loss_type == 'forced_cam':
            # similar to: http://cvrr.ucsd.edu/publications/2019/FSAFAC.pdf
            raise NotImplementedError
        elif self.args.loss_type == 'focal_attn':
            x, y = damage_loc_xy 
            max_h, max_w = attention_weights.shape[-2:]
            # TODO: NOTE: we're downscaling just for now as a hack to match CNN downscaling of feats
            # TODO: make logits/targets a 16x16 grid of "objectness" labels
            ATTN_TGT_VAL = 1
            x = x // 4
            y = y // 4

            attention_targets = torch.zeros_like(attention_weights)  # torch.zeros(max_h, max_w, dtype=torch.long).to(attention_inputs.device)
            # NOTE: skip normal images that don't have damage locations. should just be zero here.
            #import ipdb; ipdb.set_trace()
            damages_mask = torch.logical_and(x != -1, y != -1).float()
            # fill square region around (x, y) damage location for damaged images
            # TODO: batch
            for b in range(attention_targets.shape[0]):
                attention_targets[b, 
                        max(0, x[b].item() - region_width): min(max_h, x[b].item() + region_width), 
                        max(0, y[b].item() - region_width): min(max_w, y[b].item() + region_width)] = ATTN_TGT_VAL
            # apply mask
            attention_targets = attention_targets * damages_mask.unsqueeze(1).unsqueeze(2).to(device)
            return matching_losses.sigmoid_focal_loss(logits.squeeze(), target.to(logits.device).type_as(logits), reduction=reduction)
        elif self.args.loss_type == 'attn_ce':
            # 1. compute normal classification cross entropy
            ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target.to(logits.device).type_as(logits), reduction=reduction)
            # 2. compute forced attention bce
            #import ipdb; ipdb.set_trace()
            attn_weights = output['attention_weights']
            attn_ce_loss = matching_losses.BCEForcedAttentionLoss(
                    attn_weights, 
                    region_width=0,
                    damage_loc_xy=kwargs['damage_loc_xy'],
                    reduction=reduction
                )
            alpha = 1 # TODO
            return ce_loss + alpha * attn_ce_loss
        else:
            raise NotImplementedError


class MatchNet(nn.Module):
    def __init__(self, args, num_channels, num_char_classes):
        super(DamageNet, self).__init__()
        self.args = args
        self.cnn = DamageCNN(args, num_channels, use_linear_head=False)
        # self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=args.dropout)
        self.projection_net = None
        self.num_char_classes = num_char_classes
        if self.args.use_char_embedding:
            self.char_embeddings = nn.Embedding(self.num_char_classes, self.args.char_embedding_dim)
        if self.args.loss_type == 'sup_con':
            self.projection_net = ProjectionNet(
                args.cnn_output_dim + args.char_embedding_dim if args.use_char_embedding else args.cnn_output_dim, 
                args.projection_net_out_dim + args.char_embedding_dim if args.use_char_embedding else args.projection_net_out_dim, 
                args.projection_net_hidden_dim, 
                args.projection_net_type
            )
            self.criterion = SupConLoss(temperature=args.sup_con_temp, contrast_mode='all', base_temperature=args.sup_con_temp)
        # if self.args.sup_con_stage2:
        #     self.classifier = nn.Linear(args.projection_net_in_dim, 1)

    def forward(self, x, **kwargs):
        out_dict = {}
        if self.args.loss_type == 'sup_con':
            # import ipdb; ipdb.set_trace()
            augmenter = kwargs['augmenter']
            # duplicate batch
            inputs = x['image'], x['image'].clone().detach().requires_grad_(True)
            # apply diff random augmentations to each batch
            aug_inputs = (torch.zeros_like(inputs[0]), torch.zeros_like(inputs[1]))
            for batch_id in range(inputs[0].shape[0]):
                # image[batch_id] = augmenter(image[batch_id])
                aug_inputs[0][batch_id] = augmenter(inputs[0][batch_id])
                aug_inputs[1][batch_id] = augmenter(inputs[1][batch_id])
                # aug_inputs = augmenter(inputs[0]), augmenter(inputs[1])
            out_dict['aug_inputs'] = aug_inputs
            # do two separate forward passes with each batch
            out_dict['net_outputs'] = self.cnn(aug_inputs[0].to(device))['conv_flat'], self.cnn(aug_inputs[1].to(device))['conv_flat']
            proj_inputs = out_dict['net_outputs'][0], out_dict['net_outputs'][1]
            out_dict['proj_outputs'] = self.projection_net(proj_inputs[0]), self.projection_net(proj_inputs[1])
            
            return out_dict
        if self.args.sup_con_stage2:
            if ('update_encoder' in kwargs and kwargs['update_encoder']) or self.args.freeze_encoder:
                with torch.no_grad():
                    if self.args.use_template_residual:
                        template_input = kwargs['template_image'][:2].to(device) if 'template_image' in kwargs else x['avg_template'][:2].to(device)
                        template_cnn_feats = self.cnn(template_input)['conv_flat'][0].unsqueeze(0)
                        out_dict['template_cnn_feats'] = template_cnn_feats
                    cnn_feats = F.normalize(self.cnn(x['image'].to(device))['conv_flat'] - template_cnn_feats, dim=-1)
            else:
                if self.args.use_template_residual:
                    template_input = kwargs['template_image'][:2].to(device) if 'template_image' in kwargs else x['avg_template'][:2].to(device)
                    template_cnn_feats = self.cnn(template_input)[0].unsqueeze(0)['conv_flat']
                    out_dict['template_cnn_feats'] = template_cnn_feats
                cnn_feats = F.normalize(self.enc(x['image'].to(device))['conv_flat'] - template_cnn_feats, dim=-1)
        else:
            enc_feats = self.encoder(x['image'].to(device))
        out_dict['enc_feats'] = enc_feats
        import ipdb; ipdb.set_trace()
        return out_dict

    def compute_loss(self, output, target, reduction='mean', **kwargs):
        """ Use like:
        loss = model.compute_loss(out_dict, label, damage_loc_xy=damage_loc_xy)
        """
        logits = output['logits'] if 'logits' in output else None
        if self.args.loss_type == 'sup_con':
            proj1, proj2 = output['proj_outputs']
            # import ipdb; ipdb.set_trace()
            stacked_proj = torch.stack([proj1, proj2], dim=1)
            loss_pos = self.criterion(stacked_proj, eq1_loss=True)  # SimCLR loss
            loss_neg = self.criterion(stacked_proj, 1 - target)  # SupCon loss but with negs as pos
            weighted_loss_pos = self.args.sup_con_alpha * loss_pos
            weighted_loss_neg = (1 - self.args.sup_con_alpha) * loss_neg
            loss =  weighted_loss_pos + weighted_loss_neg
            if self.args.wandb:
                wandb.log({'loss_positive_term': loss_pos, 'loss_positive_term_alpha_weighted': weighted_loss_pos,
                           'loss_negative_term': loss_neg, 'loss_negative_term_alpha_weighted': weighted_loss_neg})
            return loss
        else:
            raise ValueError(f'Invalid loss type: {self.args.loss_type}')

    

class DamageNet(nn.Module):
    def __init__(self, args, num_channels, num_char_classes):
        super(DamageNet, self).__init__()
        self.args = args
        self.cnn = DamageCNN(args, num_channels, use_linear_head=False)
        # self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=args.dropout)
        self.projection_net = None
        self.num_char_classes = num_char_classes
        if self.args.use_char_embedding:
            self.char_embeddings = nn.Embedding(self.num_char_classes, self.args.char_embedding_dim)
        if self.args.loss_type == 'sup_con':
            self.projection_net = ProjectionNet(
                args.cnn_output_dim + args.char_embedding_dim if args.use_char_embedding else args.cnn_output_dim, 
                args.projection_net_out_dim + args.char_embedding_dim if args.use_char_embedding else args.projection_net_out_dim, 
                args.projection_net_hidden_dim, 
                args.projection_net_type
            )
            self.criterion = SupConLoss(temperature=args.sup_con_temp, contrast_mode='all', base_temperature=args.sup_con_temp)
        # if self.args.sup_con_stage2:
        #     self.classifier = nn.Linear(args.projection_net_in_dim, 1)
        if self.args.use_attn:
            #import ipdb; ipdb.set_trace()
            self.multihead_attn = nn.MultiheadAttention(16, num_heads=1, dropout=0.0).to(device)
        if self.args.use_rpn:
            # self.rpn = RPN(in_channels=16384, num_anchors=1)
            # 3x3 conv for predicting objectness logits
            self.object_classification_subnet = Subnet(in_channels=64, num_anchors=9, final_bias_init_pi=0.1)
            # 3x3 conv for predicting box2box transform deltas
            self.box_regression_subnet = Subnet(in_channels=64, num_anchors=9, final_bias_init_pi=None)
        
        self.final_bias_init_pi = 0.1
        
        if args.output_pooling == 'global_avg_pool_hw':
            self.fc_out = nn.Linear(in_features=64, out_features=1, bias=True) # TODO
            # nn.init.constant_(self.fc_out.bias, -np.log((1 - self.final_bias_init_pi) / self.final_bias_init_pi))
        elif args.output_pooling == 'max_pool_f':
            self.fc_out = None
        elif args.output_pooling == 'none' and args.loss_type in {'object_ce', 'focal'}:
            num_classifiers = 16 * 16  # cnn output filter is Area = height * width
            in_features = 64  # cnn features dim
            out_features = 1
            # NOTE: in_features, out_features ordering is swapped compared to nn.Linear for self.weight!
            self.clf_weight = nn.Parameter(torch.empty((num_classifiers, in_features, out_features), device=device))
            self.clf_bias = nn.Parameter(torch.empty((num_classifiers, out_features), device=device))
            # initialize like nn.Linear
            nn.init.kaiming_uniform_(self.clf_weight, a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.clf_weight[0].T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # normal linear layer bias init:
            print(f'Focal loss bias constant initialization: {-np.log((1 - self.final_bias_init_pi) / self.final_bias_init_pi):0.5f}')
            print(f'Pytorch default bias uniform range initialization: ({-bound:0.5f}, {bound:0.5f})')
            nn.init.uniform_(self.clf_bias, -bound, bound)
            # nn.init.constant_(self.clf_bias, 0.0)
            # focal loss paper bias init:
            # nn.init.constant_(self.clf_bias, -np.log((1 - self.final_bias_init_pi) / self.final_bias_init_pi))
            # focal loss paper conv layer init:
            # nn.init.normal_(self.clf_weight, std=0.01)
        elif self.args.loss_type != 'sup_con':
            self.flatten = nn.Flatten()
            self.fc_out = nn.Linear(
                args.cnn_output_dim + args.char_embedding_dim if args.use_char_embedding else args.cnn_output_dim, 
                1, 
                bias=True
            )
            # nn.init.constant_(self.fc_out.bias, -np.log((1 - self.final_bias_init_pi) / self.final_bias_init_pi))
        # self.reset_parameters()

    def reset_parameters(self):
        """ Do custom initialization of conv layers like in the Focal Loss paper """
        # All new conv layers except the final one in the RetinaNet subnets are 
        # initialized with bias b = 0 and a Gaussian weight fill with σ = 0.01.
        # import ipdb; ipdb.set_trace()
        conv_layer_num = 0
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                conv_layer_num += 1                    
                nn.init.normal_(layer.weight, std=0.01)
                # For the final conv layer of the classification subnet, 
                # we set the bias initialization to b = − log((1 − π)/π), 
                # where π specifies that at the start of training every anchor 
                # should be labeled as foreground with confidence of ∼π. 
                # We use π = .01 in all experiments, although results 
                # are robust to the exact value.
                if final_bias_init_pi is not None and conv_layer_num == len(conv_dims):
                    nn.init.constant_(layer.bias, -np.log((1 - final_bias_init_pi) / final_bias_init_pi))
                else:
                    nn.init.constant_(layer.bias, 0)
        
    def compute_cam(self, feature_conv, sigmoid_weights):
        """ Generate 1-class Class Activation Maps
        based on: https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py	
            
            Args:
                feature_conv: numpy array of shape [B, N, H, W]
                sigmoid_weights: numpy array of shape [B, 1, 64] 
        """
        if len(sigmoid_weights.shape) == 2:
            sigmoid_weights = np.expand_dims(sigmoid_weights, axis=1)
        size_upsample = (64, 64)  # TODO: original img size param
        bz, nc, h, w = feature_conv.shape
            # [B, 1, 64] . [B, 64, 16*16]
        cam = sigmoid_weights.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam = cv2.resize(cam_img, size_upsample)
        return output_cam

    def make_target_grid(self, original_res_damage_loc_xy: Tuple[int, int], original_res_grid_dim: Tuple[int, int], original_res_damage_width: int, output_grid_dim: Tuple[int, int]):
        """ Make a binarized target grid of size (1, 1, output_grid_dim[0]. output_grid_dim[1]), 
            where ones are in the max-pooled, downsampled location of damage on the convolutional 
            feature map with size grid_dim

        """
        # import ipdb; ipdb.set_trace()
        y_max, x_max = original_res_grid_dim
        original_target_grid = torch.zeros((1, 1, y_max, x_max), dtype=torch.float, device=device)
        x, y = original_res_damage_loc_xy
        d = original_res_damage_width
        original_target_grid[0, 0, max(0, y-d): min(y_max, y+d), max(0, x-d): min(x_max, x+d)] = 1.0
        target_grid = F.max_pool2d(original_target_grid, 4, stride=4)
        return target_grid
    
    def classify(self, x: torch.tensor):
        """ Classify a (B, C, H, W) cnn output feature map 
        (default is to use one classifier per location on H x W grid)
        """
        # import ipdb; ipdb.set_trace()
        if len(self.clf_weight.shape) == 2:  # clf_weight: (C, 1) where A=H*W
            # single classifier for all locations
            # TODO: collapse into batch dim and run forward pass
            pass
        elif len(self.clf_weight.shape) == 3:  # clf_weight: (A, C, 1) where A=H*W
            if len(x.shape) == 4:
                # collapse H*W to get A (area) and move area to dim 1
                x = x.view(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1).contiguous()
            assert len(x.shape) == 3, x.shape
            # import ipdb; ipdb.set_trace()
            out = torch.einsum('bac,aco->bao', x, self.clf_weight) + self.clf_bias
            return out  # (B, H*W, 1)

    def forward(self, x, **kwargs):
        out_dict = {}
        if self.args.loss_type == 'sup_con':
            # import ipdb; ipdb.set_trace()
            augmenter = kwargs['augmenter']
            # duplicate batch
            inputs = x['image'], x['image'].clone().detach().requires_grad_(True)
            # apply diff random augmentations to each batch
            aug_inputs = (torch.zeros_like(inputs[0]), torch.zeros_like(inputs[1]))
            for batch_id in range(inputs[0].shape[0]):
                # image[batch_id] = augmenter(image[batch_id])
                aug_inputs[0][batch_id] = augmenter(inputs[0][batch_id])
                aug_inputs[1][batch_id] = augmenter(inputs[1][batch_id])
                # aug_inputs = augmenter(inputs[0]), augmenter(inputs[1])
            out_dict['aug_inputs'] = aug_inputs
            # do two separate forward passes with each batch
            out_dict['net_outputs'] = self.cnn(aug_inputs[0].to(device))['conv_flat'], self.cnn(aug_inputs[1].to(device))['conv_flat']
            if self.args.use_template_residual:
                # to get around batch norm we use batch size of 2 copies hack
                template_input = kwargs['template_image'][:2].to(device) if 'template_image' in kwargs else x['avg_template'][:2].to(device)
                template_cnn_feats = self.cnn(template_input)['conv_flat'][0].unsqueeze(0)
                out_dict['template_cnn_feats'] = template_cnn_feats
                out_dict['net_outputs'] = (out_dict['net_outputs'][0] - template_cnn_feats, out_dict['net_outputs'][1] - template_cnn_feats)
            # do two separate Projections with each batch
            if self.args.use_char_embedding:
                char_embed = self.char_embeddings(torch.LongTensor(x['char_idx']).to(device))
                proj_inputs = torch.cat([out_dict['net_outputs'][0], char_embed], dim=1), torch.cat([out_dict['net_outputs'][1], char_embed], dim=1)
            else:
                proj_inputs = out_dict['net_outputs'][0], out_dict['net_outputs'][1]
            if self.args.use_char_embedding:
                proj_outputs = self.projection_net(proj_inputs[0]), self.projection_net(proj_inputs[1])
                # TODO: repeat char_embed along axes to enable cat operation
                out_dict['proj_outputs'] = torch.cat([proj_outputs[0], char_embed], dim=-1), torch.cat([proj_outputs[1], char_embed], dim=-1)
            else:
                out_dict['proj_outputs'] = self.projection_net(proj_inputs[0]), self.projection_net(proj_inputs[1])
            
            return out_dict
        if self.args.sup_con_stage2:
            if ('update_encoder' in kwargs and kwargs['update_encoder']) or self.args.freeze_encoder:
                with torch.no_grad():
                    if self.args.use_template_residual:
                        template_input = kwargs['template_image'][:2].to(device) if 'template_image' in kwargs else x['avg_template'][:2].to(device)
                        template_cnn_feats = self.cnn(template_input)['conv_flat'][0].unsqueeze(0)
                        out_dict['template_cnn_feats'] = template_cnn_feats
                    cnn_feats = F.normalize(self.cnn(x['image'].to(device))['conv_flat'] - template_cnn_feats, dim=-1)
            else:
                if self.args.use_template_residual:
                    template_input = kwargs['template_image'][:2].to(device) if 'template_image' in kwargs else x['avg_template'][:2].to(device)
                    template_cnn_feats = self.cnn(template_input)[0].unsqueeze(0)['conv_flat']
                    out_dict['template_cnn_feats'] = template_cnn_feats
                cnn_feats = F.normalize(self.cnn(x['image'].to(device))['conv_flat'] - template_cnn_feats, dim=-1)
        else:
            cnn_feats = self.cnn(x['image'].to(device))['conv']
        out_dict['cnn_feats'] = cnn_feats
        if self.args.use_rpn:
            # rpn
            region_props = self.rpn(cnn_feats)
            out_dict['region_props'] = region_props
        elif self.args.use_attn:
            # NOTE: just do max over filters now
            attn_input = cnn_feats.max(1)[0].transpose(0, 1)
            out_dict['attention_logits'], out_dict['attention_weights'] = self.multihead_attn(attn_input, attn_input, attn_input)
            out_dict['attention_logits'] = out_dict['attention_logits'].transpose(0, 1)
        #out_dict['logits'] = self.fc_out(self.dropout(self.flatten(cnn_feats)))
        # do global average pooling like Squeezenet (pool over h/w for each channel)
        if self.args.output_pooling == 'global_avg_pool_hw':
            out_dict['logits'] = self.fc_out(torch.mean(cnn_feats.view(cnn_feats.size(0), cnn_feats.size(1), -1), dim=2))
        elif self.args.output_pooling == 'max_pool_f':
            raise NotImplementedError
            # out_dict['logits'] = self.classify(torch.max(cnn_feats, dim=1))  # TODO
        elif self.args.output_pooling == 'none' and self.args.loss_type in {'object_ce', 'focal'}:
            if self.args.use_template_residual:
                template_cnn_feats = self.cnn(kwargs['template_image'][:1].to(device) if 'template_image' in kwargs else x['avg_template'][:1].to(device))
                out_dict['template_cnn_feats'] = template_cnn_feats
                cnn_feats -= template_cnn_feats
            out_dict['logits'] = self.classify(cnn_feats)
        else:
            out_dict['logits'] = self.fc_out(self.dropout(self.flatten(cnn_feats)))

        # out_dict['cam'] = self.compute_cam(cnn_feats.detach().cpu().numpy(), torch.sigmoid(out_dict['logits']).detach().cpu().numpy())
        import ipdb; ipdb.set_trace()
        return out_dict

    def compute_loss(self, output, target, reduction='mean', **kwargs):
            #loss = model.compute_loss(out_dict, label, damage_loc_xy=damage_loc_xy)
        logits = output['logits'] if 'logits' in output else None
        if self.args.loss_type == 'sup_con':
            proj1, proj2 = output['proj_outputs']
            # import ipdb; ipdb.set_trace()
            stacked_proj = torch.stack([proj1, proj2], dim=1)
            loss_pos = self.criterion(stacked_proj, eq1_loss=True)  # SimCLR loss
            loss_neg = self.criterion(stacked_proj, 1 - target)  # SupCon loss but with negs as pos
            # print(f'Damage Similarity Loss: {loss_pos.item()}')
            # print(f'Normal Similarity Loss: {loss_neg.item()}')
            weighted_loss_pos = self.args.sup_con_alpha * loss_pos
            weighted_loss_neg = (1 - self.args.sup_con_alpha) * loss_neg
            loss =  weighted_loss_pos + weighted_loss_neg
            if self.args.wandb:
                wandb.log({'loss_positive_term': loss_pos, 'loss_positive_term_alpha_weighted': weighted_loss_pos,
                           'loss_negative_term': loss_neg, 'loss_negative_term_alpha_weighted': weighted_loss_neg})
            return loss
            #target # should be [bsz] int labels (damage or normal)    
        elif self.args.loss_type == 'ce':
            ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target.to(logits.device).type_as(logits), reduction=reduction)
            return ce_loss
        # elif self.args.loss_type == 'sup_contrast':
        elif self.args.loss_type == "object_ce":
            xy_targets = []
            xy_logits = []
            # TODO: get one negative box grid off char skeleton and use in loss to make it balanced!
            xys = [
                (x.item(), y.item()) 
                for x, y in zip(*kwargs['damage_loc_xy'])
            ]
            non_dam_xys = [
                (x.item(), y.item()) 
                for x, y in zip(*kwargs['non_damage_loc_xy'])
            ]
            # import ipdb; ipdb.set_trace()
            for b in range(len(xys)):
                x, y = xys[b]
                if x == -1 or y == -1:
                    # NOTE: skip negative examples!
                    continue
                    xy_logits.append(logits[b])
                    xy_targets.append(torch.zeros((1, 1, 16, 16), dtype=torch.float, device=device).view(-1))
                else:
                    xy_logits.append(logits[b])
                    # import ipdb; ipdb.set_trace()
                    xy_targets_pos = self.make_target_grid(xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1)
                    xy_targets_neg = self.make_target_grid(non_dam_xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1)
                    # xy_targets.append((xy_targets_pos, xy_targets_neg))
                    xy_targets.append(xy_targets_pos)
                    
            xy_targets = torch.stack(xy_targets)
            xy_logits = torch.stack(xy_logits)
            # NOTE: only compute loss on positive examples or what?
            # import ipdb; ipdb.set_trace()
            # pos_weight = torch.full((1,), (1-xy_targets).sum() / xy_targets.sum()).to(device)
            pos_weight = self.args.pos_weight_ce_multiplier * torch.full((1,), (1-xy_targets).sum() / xy_targets.sum()).to(device) if self.args.use_pos_weight_on_ce and xy_targets.sum() > 0.0 else None
            # ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), xy_targets.to(logits.device), reduction=reduction, pos_weight=pos_weight)
            ce_loss = F.binary_cross_entropy_with_logits(xy_logits.squeeze(), xy_targets.to(logits.device), reduction=reduction, pos_weight=pos_weight)


            # import ipdb; ipdb.set_trace()
            return ce_loss
        elif self.args.loss_type == 'focal':
            xy_targets = []
            xys = [(x.item(), y.item()) for x, y in zip(*kwargs['damage_loc_xy'])]
            for b in range(target.shape[0]):
                x, y = xys[b]
                if x == -1 or y == -1:
                    xy_targets.append(torch.zeros((1, 1, 16, 16), dtype=torch.float, device=device).view(-1))
                else:
                    xy_targets.append(self.make_target_grid(xys[b], (64, 64), original_res_damage_width=4, output_grid_dim=(16, 16)).view(-1))
            # import ipdb; ipdb.set_trace()
            xy_targets = torch.stack(xy_targets)
            return matching_losses.sigmoid_focal_loss(logits.squeeze(), xy_targets.to(logits.device).type_as(logits), reduction=reduction)
        elif self.args.loss_type == 'forced_cam':
            # similar to: http://cvrr.ucsd.edu/publications/2019/FSAFAC.pdf
            raise NotImplementedError
        elif self.args.loss_type == 'focal_attn':
            x, y = damage_loc_xy 
            max_h, max_w = attention_weights.shape[-2:]
            # TODO: NOTE: we're downscaling just for now as a hack to match CNN downscaling of feats
            # TODO: make logits/targets a 16x16 grid of "objectness" labels
            ATTN_TGT_VAL = 1
            x = x // 4
            y = y // 4

            attention_targets = torch.zeros_like(attention_weights)  # torch.zeros(max_h, max_w, dtype=torch.long).to(attention_inputs.device)
            # NOTE: skip normal images that don't have damage locations. should just be zero here.
            #import ipdb; ipdb.set_trace()
            damages_mask = torch.logical_and(x != -1, y != -1).float()
            # fill square region around (x, y) damage location for damaged images
            # TODO: batch
            for b in range(attention_targets.shape[0]):
                attention_targets[b, 
                        max(0, x[b].item() - region_width): min(max_h, x[b].item() + region_width), 
                        max(0, y[b].item() - region_width): min(max_w, y[b].item() + region_width)] = ATTN_TGT_VAL
            # apply mask
            attention_targets = attention_targets * damages_mask.unsqueeze(1).unsqueeze(2).to(device)
            return matching_losses.sigmoid_focal_loss(logits.squeeze(), target.to(logits.device).type_as(logits), reduction=reduction)
        elif self.args.loss_type == 'attn_ce':
            # 1. compute normal classification cross entropy
            ce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target.to(logits.device).type_as(logits), reduction=reduction)
            # 2. compute forced attention bce
            #import ipdb; ipdb.set_trace()
            attn_weights = output['attention_weights']
            attn_ce_loss = matching_losses.BCEForcedAttentionLoss(
                    attn_weights, 
                    region_width=0,
                    damage_loc_xy=kwargs['damage_loc_xy'],
                    reduction=reduction
                )
            alpha = 1 # TODO
            return ce_loss + alpha * attn_ce_loss
        else:
            raise NotImplementedError
