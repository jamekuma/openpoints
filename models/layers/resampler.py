import torch
import torch.nn as nn
from ..build import MODELS
from openpoints.utils.chamfer_distance import ChamferDistance

@MODELS.register_module()
class Resampler(nn.Module):
    """
    Basic Resampler, move loss is mean
    """
    def __init__(self, 
        in_channel=3, 
        mlp_channels=[128,128], 
        mlp_ds_1_channels=[128,128], 
        mlp_ds_2_channels=[128,128], 
        out_channel=3,
        **kwargs,
    ):
        super().__init__()
        self.mlp = []
        last_c = in_channel
        for c in mlp_channels:
            self.mlp.append(nn.Linear(last_c, c))
            self.mlp.append(nn.ReLU(inplace=True))
            last_c = c
        self.mlp = nn.Sequential(*self.mlp)

        self.mlp_ds_1 = []
        last_c = 3 + mlp_channels[-1]
        for c in mlp_ds_1_channels:
            self.mlp_ds_1.append(nn.Linear(last_c, c))
            self.mlp_ds_1.append(nn.ReLU(inplace=True))
            last_c = c
        self.mlp_ds_1 = nn.Sequential(*self.mlp_ds_1)
        
        self.mlp_ds_2 = []
        last_c = 3 + mlp_ds_1_channels[-1]
        for c in mlp_ds_2_channels:
            self.mlp_ds_2.append(nn.Linear(last_c, c))
            self.mlp_ds_2.append(nn.ReLU(inplace=True))
            last_c = c
        self.mlp_ds_2.append(nn.Linear(last_c, out_channel))
        self.mlp_ds_2 = nn.Sequential(*self.mlp_ds_2)
        self.loss = torch.Tensor([0]).cuda()

        self.cd_loss_coef = kwargs.get("cd_loss_coef", 0)
        self.move_loss_coef = kwargs.get('move_loss_coef', 0)
        self.cd_with = kwargs.get("cd_with", "recent")
        self.input_type = kwargs.get("input_type", "recent")
        # print(f'move_loss_coef = {self.move_loss_coef}')

        self.ori_pos = None     # record the origin points
        self.ori_emb_pos = None     # record the origin points global feat

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def ra_pooling(self, feat: torch.Tensor, origin_num : int):
        """_summary_

        Args:
            feat: [B, N, C]

        Returns:
            [B, 1, C]
        """
        return torch.sum(feat, dim=1, keepdim=True) / origin_num
    
    def update_loss(self, pos, new_pos_ds, delta_pos_ds):
        """_summary_

        Args:
            pos: [B, N, 3]
            new_pos_ds: [B, S, 3]
            delta_pos_ds: [B, S, 3]
        """
        N = pos.shape[1]
        S = new_pos_ds.shape[1]

        # move loss
        move_len = torch.sqrt(torch.sum(delta_pos_ds ** 2, dim=-1))   # [B, S]
        sum_move_len = torch.sum(move_len, dim=-1) / N
        move_loss = torch.mean(sum_move_len)

        # cd loss
        if self.cd_with == "recent":
            target_pos = pos
        elif self.cd_with == "origin":
            target_pos = self.ori_pos
        else:
            raise NotImplementedError(f"self.cd_with = {self.cd_with} is invalid")
        cd_loss = self.__cd_loss(target_pos, new_pos_ds)

        loss_one_stage = move_loss * self.move_loss_coef + cd_loss * self.cd_loss_coef

        self.loss += loss_one_stage
        
    def __cd_loss(self, pos, pos_ds, gamma=1, delta=0):
        cost_p1_p2, cost_p2_p1 = ChamferDistance()(pos_ds, pos)
        pc_size = pos_ds.shape[1]
        max_cost = torch.max(cost_p1_p2, dim=1)[0]  # furthest point
        max_cost = torch.mean(max_cost)
        cost_p1_p2 = torch.mean(cost_p1_p2)
        cost_p2_p1 = torch.mean(cost_p2_p1)
        loss = cost_p1_p2 + max_cost + (gamma + delta * pc_size) * cost_p2_p1
        return loss
    # def init_loss(self):
    #     self.loss = 0

    def get_loss(self):
        ret = self.loss
        self.loss = torch.Tensor([0]).cuda()
        self.ori_pos = None
        self.ori_emb_pos = None
        return ret

    def forward(self, pos, pos_ds):
        """_summary_

        Args:
            pos: xyz [B, N, 3]
            pos_ds: xyz [B, S, 3] (S < N)
        """
        if self.ori_pos is None:
            self.ori_pos = pos
        B, N, _ = pos.shape
        _, S, _ = pos_ds.shape
        if self.input_type == "recent":
            emb_pos = torch.max(self.mlp(pos), dim=1, keepdim=True)[0]                      # [B, N, 3] -mlp-> [B, N, C] -max-> [B, 1, C]
        elif self.input_type == "origin":
            if self.ori_emb_pos is None:
                self.ori_emb_pos = torch.max(self.mlp(self.ori_pos), dim=1, keepdim=True)[0]
            emb_pos = self.ori_emb_pos
        emb_pos_ds = self.ra_pooling(self.mlp_ds_1(torch.cat([pos_ds, emb_pos.expand(-1, S, -1)], dim=2)), N)       # [B, S, (C + 3)] -mlp-> [B, S, C_1] -ra_pooling-> [B, 1, C_1]
        delta_pos_ds = self.mlp_ds_2(torch.cat([emb_pos_ds.expand(-1, S, -1), pos_ds], dim=2))          # [B, S, (C_1 + 3)] -mlp-> [B, S, 3]
        new_pos_ds = pos_ds + delta_pos_ds
        self.update_loss(pos, new_pos_ds, delta_pos_ds)
        return new_pos_ds, delta_pos_ds

@MODELS.register_module()
class Resampler_v2(Resampler):
    def update_loss(self, delta_pos_ds, N, S):
        move_len = torch.sqrt(torch.sum(delta_pos_ds ** 2, dim=-1))   # [B, S]
        sum_move_len = torch.sum(move_len, dim=-1) / N
        self.loss = torch.max(self.loss, torch.mean(sum_move_len))