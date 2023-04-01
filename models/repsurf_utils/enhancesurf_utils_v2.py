"""
enhance_surf_v2
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from .pointnet2_utils import query_knn_point, index_points

class EnhanceSurfaceConstructor(nn.Module):
    """
    Umbrella-based Surface Abstraction Module

    """

    def __init__(self, k, in_channel, aggr_type='sum', random_inv=True, weighted=False, temperature=1):
        super(EnhanceSurfaceConstructor, self).__init__()
        self.k = k
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.weighted = weighted
        self.temperature = temperature

        self.mlps = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            # nn.Linear(in_channel, in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            # nn.Linear(in_channel, in_channel, bias=True),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 1, bias=True),
            # nn.Linear(in_channel, in_channel, bias=True),
        )

    def forward(self, center):
        """_summary_

        Args:
            center: original points [B, N, 3]

        Returns:
            surf_feat: [B, C, N]
        """
        surface_feat = get_surface_feat(center, self.k, self.random_inv, self.weighted, self.temperature)    # [B, N, K, C]
        surface_feat = surface_feat.permute(0, 3, 2, 1)                     # [B, C, K, N]
        surface_feat = self.mlps(surface_feat)                              # [B, C, K, N]
        # aggregation
        if self.aggr_type == 'max':
            new_feature = torch.max(surface_feat, 2)[0]
        elif self.aggr_type == 'avg':
            new_feature = torch.mean(surface_feat, 2)
        else:
            new_feature = torch.sum(surface_feat, 2)
        return new_feature                                                  # [B, C, N]


def get_weight(X, temperature=1):
    """_summary_

    Args:
        X: Batch of of points, every group has K points, [B, K, 3]
    """
    dist_square = torch.sum(X ** 2, dim=-1)       # [B, K]
    exp_weight = torch.exp(-dist_square / (temperature ** 2))       # [B, K]
    norm_exp_weight = exp_weight / torch.sum(exp_weight, dim=-1, keepdim=True)    # [B, K]
    return torch.sqrt(norm_exp_weight.unsqueeze(-1))


def cov_analyze(X, method='eigh', weighted=False, temperature=1):
    """_summary_

    Args:
        X: Batch of N group of points, every group has K points, [B, N, K, K, 3]
    """
    B, N, _, K, _ = X.shape
    X = torch.flatten(X, start_dim=0, end_dim=2)        # [B*N*K, K, 3]
    if weighted:
        weight = get_weight(X, temperature)
        X = X * weight
    # u, s, v = torch.svd(X)
    if method == 'svd':
        u, s, v = torch.linalg.svd(X)
        lambda_1, lambda_2, lambda_3 = s[:, 0] ** 2, s[:, 1] ** 2, s[:, 2] ** 2 # lambda_1 > lambda_2 > lambda_3
        v_1, v_2, v_3 = v[:, 0, :], v[:, 1, :], v[:, 2, :]
    elif method == 'eigh':
        cov = torch.bmm(X.permute(0, 2, 1), X)
        L, Q = torch.linalg.eigh(cov)
        lambda_1, lambda_2, lambda_3 = L[:, 2], L[:, 1], L[:, 0]   # lambda_1 > lambda_2 > lambda_3
        v_1, v_2, v_3 = Q[:, :, 2], Q[:, :, 1], -Q[:, :, 0]
    else:
        raise NotImplementedError(f'{method} is not implemented')

    lambda_1 = lambda_1 / v_1.norm(p=2, dim=1)
    lambda_2 = lambda_2 / v_2.norm(p=2, dim=1)
    lambda_3 = lambda_3 / v_3.norm(p=2, dim=1)
    v_1 = v_1 / v_1.norm(p=2, dim=1, keepdim=True)
    v_2 = v_2 / v_2.norm(p=2, dim=1, keepdim=True)
    v_3 = v_3 / v_3.norm(p=2, dim=1, keepdim=True)


    a = (lambda_1 - lambda_2) / lambda_1    # linearity
    p = (lambda_2 - lambda_3) / lambda_1    # planarity
    s = lambda_3 / lambda_1                 # sphericity
    return a.view(B, N, K, 1), p.view(B, N, K, 1), s.view(B, N, K, 1), v_1.view(B, N, K, 3), v_2.view(B, N, K, 3), v_3.view(B, N, K, 3)

def get_surface_feat(points, k=8, random_inv=False, weighted=False, temperature=1):
    """_summary_

    Args:
        points: original points [B, N, 3]
        k: number of local points. Defaults to 8.
    return:
        [B, N, k, C]
    """
    # points = points.permute(0, 2, 1)
    idx = query_knn_point(k, points, points, cuda=True)  # [B, N, k]
    torch.cuda.empty_cache()
    group_points = index_points(points, idx, cuda=True, is_group=True)  # [B, N, k, 3]
    torch.cuda.empty_cache()
    group_group_points = group_points.unsqueeze(-3).repeat(1, 1, k, 1, 1)       # [B, N, k, k, 3]
    group_group_points_norm = group_group_points - group_points.unsqueeze(-2)   # [B, N, k, k, 3]
    # group_points_norm = group_points - points.unsqueeze(-2)
    # group_points_norm = group_points - group_points.mean(dim=2, keepdim=True)

    a, p, s, v_1, v_2, v_3 = cov_analyze(group_group_points_norm, weighted=weighted, temperature=temperature)
    if random_inv:
        random_mask_1 = torch.randint(0, 2, (v_1.size(0), 1, 1, 1)).float() * 2. - 1.
        random_mask_2 = torch.randint(0, 2, (v_1.size(0), 1, 1, 1)).float() * 2. - 1.
        random_mask_1 = random_mask_1.cuda()
        random_mask_2 = random_mask_2.cuda()
        v_1 = v_1 * random_mask_1
        v_2 = v_2 * random_mask_2
        # v_3 = v_3 * random_mask
        v_3 = torch.cross(v_1, v_2, dim=-1)
    return torch.cat((a, p, s, v_1, v_2, v_3), dim=-1)  # (B, N, K, 12)