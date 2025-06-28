import torch
import torch.nn as nn
from einops import rearrange
import math

__all__ = (
    "ConvDu",
    "StackedAttentionBlock"
)


############################################ ConvDu ######################################################################
class ConvDu(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        
        self.in_channels =  c1
        self.out_channels =  c2
        self.kernel_size = k
        self.stride = s

        convs = []

        if self.stride == 1:
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=self.in_channels))
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=self.in_channels))
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=self.in_channels))

        elif self.stride == 2:
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=2, padding=1, groups=self.in_channels))
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=self.in_channels))
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=self.in_channels))
            convs.append(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=1, padding=1, groups=self.in_channels))

        
        self.ChannelShuffle = ChannelShuffle(4)

            
        convs.append(nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, bias=False, groups=4))


        self.convs = nn.ModuleList(convs)

        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):

        if self.stride == 1:
            temp_1 = x
            temp_2 = self.convs[0](temp_1)
            temp_3 = self.convs[1](temp_2)
            temp_4 = self.convs[2](temp_3)
            out = temp_1 + temp_2 + temp_3 + temp_4
            out = self.convs[3](out)

        elif self.stride == 2:
            temp_1 = self.convs[0](x)
            temp_2 = self.convs[1](temp_1)
            temp_3 = self.convs[2](temp_2)
            temp_4 = self.convs[3](temp_3)
            out = temp_1 + temp_2 + temp_3 + temp_4
            out = self.convs[4](out)

        else:
            raise ValueError(f"Unsupported stride value: {self.stride}")

        out = self.ChannelShuffle(out)
        return self.act(self.bn(out))

"""
    def forward_fuse(self, x):
        input = x
        spx = torch.split(x, 4, 1)
        for i in range(4):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        x = torch.cat((out, spx[4]), 1)
        return self.act(x)
"""

############################################ LowRankGuidedAttention ######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankGuidedAttention(nn.Module):
    def __init__(self, in_channels, rank_channels=8, attn_dropout=0.0):
        super().__init__()
        self.guidance_proj = nn.Conv2d(in_channels, rank_channels, kernel_size=1, bias=True)
        self.key_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        g = self.guidance_proj(x)                           # shape: (B, r, H, W)
        g_mean = F.adaptive_avg_pool2d(g, output_size=1)    # shape: (B, r, 1, 1)

        k = self.key_proj(x)                                # shape: (B, C, H, W)
        v = self.value_proj(x)

        attn_score = torch.sigmoid(torch.sum(k * g_mean, dim=1, keepdim=True))  # (B, 1, H, W)
        attn_score = self.dropout(attn_score)

        out = v * attn_score                                # Apply attention
        out = self.out_proj(out)

        return out


class LowRankAttnFFN(nn.Module):
    def __init__(self, embed_dim, ffn_latent_dim, attn_dropout=0.0, dropout=0.0):
        super().__init__()
        # Attention 分支
        self.pre_norm_attn = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=embed_dim),
            LowRankGuidedAttention(embed_dim, rank_channels=8, attn_dropout=attn_dropout),
            nn.Dropout(dropout)
        )

        # FFN 分支
        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=embed_dim),
            nn.Conv2d(embed_dim, ffn_latent_dim, kernel_size=1, bias=True),
            nn.SiLU(),  # 激活函数
            nn.Dropout(dropout),
            nn.Conv2d(ffn_latent_dim, embed_dim, kernel_size=1, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 注意力残差
        x = x + self.pre_norm_attn(x)
        # FFN 残差
        x = x + self.pre_norm_ffn(x)
        return x


############################################ 分组注意力机制 ######################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyCustomAttentionHead(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x_flat):  # [B, HW, C]
        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)
        attn = torch.softmax(Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5), dim=-1)
        out = attn @ V
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)

class StackedAttentionBlock(nn.Module):
    def __init__(self, in_channels, height, width, num_heads=4, ffn_dropout=0.1):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.chunk_channels = in_channels // num_heads
        self.height = height
        self.width = width

        self.attn_heads = nn.ModuleList([
            MyCustomAttentionHead(self.chunk_channels, height, width)
            for _ in range(num_heads)
        ])
        self.attn_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.norm1 = nn.LayerNorm(in_channels)

        self.ffn = FeedForward(in_channels, dropout=ffn_dropout)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x

        # Step 1: Attention
        chunks = torch.chunk(x, self.num_heads, dim=1)  # [B, C/4, H, W]
        outputs = []
        accumulated = torch.zeros_like(chunks[0])

        for i in range(self.num_heads):
            xi = chunks[i] + accumulated
            xi_flat = xi.flatten(2).transpose(1, 2)  # [B, HW, C]
            out_flat = self.attn_heads[i](xi_flat)
            out = out_flat.transpose(1, 2).reshape(B, self.chunk_channels, H, W)
            outputs.append(out)
            accumulated = out

        x_attn = torch.cat(outputs, dim=1)  # [B, C, H, W]
        x_attn = self.attn_proj(x_attn)

        # Step 2: Residual + Norm
        x = residual + x_attn
        x_reshaped = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x_norm1 = self.norm1(x_reshaped)

        # Step 3: FFN
        x_ffn = self.ffn(x_norm1)
        x = x_norm1 + x_ffn
        x = self.norm2(x)

        # Reshape back to [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
    

############################################  低秩 Attention 投影（Q/K降维） ######################################################################
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise ImportError("This module requires PyTorch. Please install it with 'pip install torch'.") from e


class LowRankAttentionHead(nn.Module):
    def __init__(self, in_channels, rank):
        super().__init__()
        self.in_channels = in_channels
        self.rank = rank

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)

        self.q_low = nn.Linear(in_channels, rank, bias=False)
        self.k_low = nn.Linear(in_channels, rank, bias=False)
        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, HW, C]
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)

        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)

        Q_low = self.q_low(Q)  # [B, HW, r]
        K_low = self.k_low(K)  # [B, HW, r]

        attn = torch.matmul(Q_low, K_low.transpose(-2, -1)) / (self.rank ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # [B, HW, C]
        out = self.out_proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class MultiHeadHierarchicalAttentionBlock(nn.Module):
    def __init__(self, in_channels, rank=16, dropout=0.1):
        super().__init__()
        assert in_channels % 4 == 0, "in_channels must be divisible by 4"
        split_c = in_channels // 4

        self.head1 = LowRankAttentionHead(split_c, rank)
        self.head2 = LowRankAttentionHead(split_c, rank)
        self.head3 = LowRankAttentionHead(split_c, rank)
        self.head4 = LowRankAttentionHead(split_c, rank)

        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        split_c = C // 4
        x1, x2, x3, x4 = torch.split(x, split_c, dim=1)

        out1 = self.head1(x1)
        out2 = self.head2(x2 + out1)
        out3 = self.head3(x3 + out2)
        out4 = self.head4(x4 + out3)

        concat = torch.cat([out1, out2, out3, out4], dim=1)
        out = x + self.norm1(concat)
        out = out + self.norm2(self.ffn(out))
        return out



############################################  MultiScaleAttentionBlock ######################################################################






class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.size()
        g = self.groups
        x = x.view(B, g, C // g, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class SimpleAttentionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1, batch_first=True)

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
        x, _ = self.attn(x, x, x)
        return x

class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_channels, groups=4):
        super().__init__()
        assert in_channels % groups == 0, "Input channels must be divisible by number of groups"
        self.groups = groups
        self.chunk = in_channels // groups

        self.res_convs = nn.ModuleList([
            ResidualConvUnit(self.chunk) for _ in range(groups - 1)
        ])

        self.group_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups)
        self.shuffle = ChannelShuffle(groups=groups)

        self.attn_heads = nn.ModuleList([
            SimpleAttentionHead(self.chunk) for _ in range(groups)
        ])
        self.flatten = nn.Flatten(2)  # B, C, H, W → B, C, H*W

    def forward(self, x):
        B, C, H, W = x.size()
        parts = torch.chunk(x, self.groups, dim=1)
        out_parts = [parts[0]]

        prev = parts[0]
        for i in range(self.groups - 1):
            current = parts[i + 1]
            added = current + prev
            out = self.res_convs[i](added)
            out_parts.append(out)
            prev = out

        out = torch.cat(out_parts, dim=1)
        out = self.group_conv(out)
        out = self.shuffle(out)

        chunks = torch.chunk(out, self.groups, dim=1)
        attn_outs = []
        residual = None

        for i in range(self.groups):
            flat = self.flatten(chunks[i]).transpose(1, 2)  # (B, H*W, C)
            attn = self.attn_heads[i](flat, residual)
            residual = flat  # pass to next head
            attn_out = attn.transpose(1, 2).view(B, self.chunk, H, W)
            attn_outs.append(attn_out)

        return torch.cat(attn_outs, dim=1)

# Test Case
if __name__ == "__main__":
    x = torch.randn(2, 64, 16, 16)  # Batch size 2, 64 channels, 16x16 feature map
    block = MultiHeadHierarchicalAttentionBlock(in_channels=64, rank=8)
    out = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "Output shape mismatch!"
    print("MultiHeadHierarchicalAttentionBlock Test passed.")

    model = StackedAttentionBlock(in_channels=64, height=32, width=32)
    x  = torch.randn(8, 64, 32, 32)
    out = model(x)  # shape: [8, 64, 32, 32]
    print("StackedAttentionBlock Test passed.")
