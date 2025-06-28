import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = (
    "CustomConvAttentionBlock"
)



##################################################- 卷积部分 -############################################################################


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        x = x.view(B, g, C // g, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)


##################################################- 注意力机制部分 -############################################################################
import torch
import torch.nn as nn
from einops import rearrange

# ------------------------- 基础模块 -------------------------

# 深度可分离卷积模块（假设你已有定义）
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
    def forward(self, x):
        return self.conv(x)
    

def conv_1x1_group_bn(inp, oup, groups):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=1, groups=groups, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

# 通道洗牌模块
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

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
# 前馈网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 单头注意力模块
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 计算 Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        # 注意力打分
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        # 聚合 V
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)

# --------------------- 多尺度注意力模块 ----------------------

class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_head, groups):
        super().__init__()
        self.groups = groups
        self.group_channels = dim // groups
        self.ph, self.pw = 2, 2  # Patch 高宽

        # 对整个输入进行 LayerNorm
        self.norm = nn.LayerNorm(dim)

        # 为每个 group 构建 Attention + FFN 子模块（2层）
        self.attention = nn.ModuleList([
            nn.ModuleList([
                Attention(dim // groups, 1, dim // (2*groups)),
                FeedForward(dim // groups, (dim*2)// groups, dropout=0.0)
            ]) for _ in range(groups)
        ])

    def forward(self, x):
        B, C, H, W = x.shape

        # 重排成 4D attention 格式: [B, P, Seq_len, D]
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.norm(x)  # 局部归一化
        # 通道维度分组，按 dim 分为多个 group
        chunks = torch.chunk(x, self.groups, dim=3)
        outputs = []
        out = chunks[0]

        # 残差式注意力链：每组输入为前一组输出+当前输入
        for i in range(self.groups):
            attn, ffn = self.attention[i]
            out = attn(out)
            out = ffn(out)

            outputs.append(out)
            if i < self.groups - 1:
                out = chunks[i + 1] + out  # 残差连接

        # 合并所有 group 的输出
        out = torch.cat(outputs, dim=3)

        # 恢复原图像形状
        out = rearrange(out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)',
                        h=H // self.ph, w=W // self.pw, ph=self.ph, pw=self.pw)

        return out

# -------------------- 主模块：卷积 + 注意力 ----------------------

class CustomConvAttentionBlock(nn.Module):
    def __init__(self, in_channels, dim, depth, groups=4):
        super().__init__()
        assert in_channels % groups == 0, "in_channels must be divisible by group count"

        self.groups = groups
        self.group_channels = in_channels // groups

        # 多组深度可分离卷积，用于构建多尺度特征
        self.depthwise = nn.ModuleList([
            DepthwiseConv(in_channels) for _ in range(groups - 1)
        ])

        # 通道洗牌
        self.shuffle = ChannelShuffle(groups=groups)

        # 1x1分组卷积：维度压缩
        self.reduce_conv = conv_1x1_group_bn(
            in_channels * groups, dim, groups=1
        )

        # 多层多尺度注意力模块
        self.multi_scale_attention = nn.ModuleList([
            MultiScaleAttention(dim, 16, groups) for _ in range(depth)
        ])

        # 最终输出恢复维度
        self.out_conv = conv_nxn_bn(dim, in_channels, kernal_size=3)

    def forward(self, x):
        B, C, H, W = x.shape

        # 残差卷积链：x → r1 → r2 → r3
        residuals = [x]
        for dw in self.depthwise:
            residuals.append(dw(residuals[-1]))

        # 拼接多个残差输出
        merged = torch.cat(residuals, dim=1)  # [B, 4*C, H, W]
        shuffled = self.shuffle(merged)

        # 通道降维
        reduced = self.reduce_conv(shuffled)  # [B, dim, H, W]


        # 多层 Attention
        out = reduced
        for attn in self.multi_scale_attention:
            out = attn(out)

        # 输出恢复维度
        out = self.out_conv(out)
        return out

# ------------------------ 测试代码 ------------------------

if __name__ == "__main__":
    x = torch.randn(2, 64, 16, 16)  # 测试输入
    block = CustomConvAttentionBlock(in_channels=64, dim=96, depth=4, groups=4)
    out = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    assert out.shape == x.shape, "Output shape mismatch!"
    print("CustomConvAttentionBlock Test passed.")
