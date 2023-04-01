import torch
import torch.nn as nn

class Resampler(nn.Module):
    def __init__(self, in_channel=3, mlp_channels=[128,128], mlp_ds_1_channels=[128,128], mlp_ds_2_channels=[128,128], out_channel=3):
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
        self.loss = 0
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
    
    def update_loss(self, delta_pos_ds, N, S):
        move_len = torch.sqrt(torch.sum(delta_pos_ds ** 2, dim=-1))   # [B, S]
        sum_move_len = torch.sum(move_len, dim=-1) / N
        self.loss += torch.mean(sum_move_len)
        

    # def init_loss(self):
    #     self.loss = 0

    def get_loss(self):
        ret = self.loss
        self.loss = 0
        return ret

    def forward(self, pos, pos_ds):
        """_summary_

        Args:
            pos: xyz [B, N, 3]
            pos_ds: xyz [B, S, 3] (S < N)
        """
        B, N, _ = pos.shape
        _, S, _ = pos_ds.shape
        emb_pos = torch.max(self.mlp(pos), dim=1, keepdim=True)[0]                      # [B, N, 3] -mlp-> [B, N, C] -max-> [B, 1, C]
        emb_pos_ds = self.ra_pooling(self.mlp_ds_1(torch.cat([pos_ds, emb_pos.expand(-1, S, -1)], dim=2)), N)       # [B, S, (C + 3)] -mlp-> [B, S, C_1] -ra_pooling-> [B, 1, C_1]
        delta_pos_ds = self.mlp_ds_2(torch.cat([emb_pos_ds.expand(-1, S, -1), pos_ds], dim=2))          # [B, S, (C_1 + 3)] -mlp-> [B, S, 3]
        pos_ds = pos_ds + delta_pos_ds
        self.update_loss(delta_pos_ds, N, S)
        return pos_ds, delta_pos_ds