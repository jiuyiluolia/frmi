import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, channel, time_len, roi_num, domain):
        super().__init__()
        self.channel = channel
        self.time_len = time_len
        self.roi_num = roi_num
        
        # 生成位置编码
        if domain == "spatial":
            positions = torch.arange(roi_num).float().unsqueeze(1)
        else:  # temporal
            positions = torch.arange(time_len).float().unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, channel, 2).float() * (-np.log(10000.0) / channel))
        pe = torch.zeros(roi_num if domain == "spatial" else time_len, channel)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        
        if domain == "spatial":
            self.pe = pe.unsqueeze(0).unsqueeze(0).repeat(1, time_len, 1, 1)  # (1, T, V, C)
        else:
            self.pe = pe.unsqueeze(0).unsqueeze(2).repeat(1, 1, roi_num, 1)  # (1, T, V, C)
    
    def forward(self, x):
        # x: (N, C, T, V)
        x = x.permute(0, 2, 3, 1)  # (N, T, V, C)
        x = x + self.pe.to(x.device)
        return x.permute(0, 3, 1, 2)  # (N, C, T, V)

class STAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, roi_num, time_len, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.roi_num = roi_num
        
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, roi_num, 1)
        )
        
        # 时间卷积
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        # x: (N, C, T, V)
        residual = self.residual(x)
        
        # 空间注意力
        att = self.spatial_att(x)  # (N, V, T, V)
        att = torch.softmax(att, dim=-1)
        x_att = torch.einsum('nctv,nvtq->nctq', x, att)
        
        # 时间卷积
        out = self.temporal_conv(x_att)
        out = out + residual  # 残差连接
        
        return out

class SyncAsyncBrainNet(nn.Module):
    def __init__(self, num_roi=90, num_time=128, num_class=2):
        super().__init__()
        self.num_roi = num_roi
        self.num_time = num_time
        
        # 输入映射
        self.input_map = nn.Conv2d(1, 16, (3, 1), padding=(1, 0))
        
        # 位置编码
        self.spatial_pe = PositionalEncoding(16, num_time, num_roi, "spatial")
        self.temporal_pe = PositionalEncoding(16, num_time, num_roi, "temporal")
        
        # 注意力块
        self.att_block1 = STAttentionBlock(16, 32, num_roi, num_time)
        self.att_block2 = STAttentionBlock(32, 64, num_roi, num_time//2, kernel_size=3)
        
        # 全局池化和分类
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_class)
    
    def forward(self, x):
        # x: (N, 1, T, V, 1)
        x = x.squeeze(-1)  # (N, 1, T, V)
        
        # 输入映射
        x = self.input_map(x)
        
        # 位置编码
        x = self.spatial_pe(x) + self.temporal_pe(x)
        
        # 注意力块
        x = self.att_block1(x)
        x = nn.functional.max_pool2d(x, (2, 1))  # 时间维度下采样
        x = self.att_block2(x)
        
        # 分类
        x = self.pool(x).squeeze()
        x = self.fc(x)
        
        return x
