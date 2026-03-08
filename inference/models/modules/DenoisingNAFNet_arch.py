import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from .local_arch import Local_Base
from .module_util import SinusoidalPosEmb, LayerNorm, exists
import matplotlib.pyplot as plt
import seaborn as sns
def extract_patches(tensor, patch_size, stride=1):
    """
    使用 unfold 提取 tensor 的所有patch。
    :param tensor: 输入特征 (B, C, H, W)
    :param patch_size: patch 的大小 (ph, pw)
    :param stride: patch 提取的步长
    :return: 提取的 patches (B, C*ph*pw, num_patches)
    """
    B, C, H, W = tensor.shape
    ph, pw = patch_size
    patches = F.unfold(tensor, kernel_size=(ph, pw), stride=stride)  # (B, C*ph*pw, num_patches)
    return patches

def knn_patches(foreground, background, k=1):
    """
    找到背景中与前景最相似的 top-k patch。
    :param foreground: 前景特征 (B, C, H, W)
    :param background: 背景特征 (B, C, H_bg, W_bg)
    :param patch_size: patch 的大小 (ph, pw)
    :param k: 找到的 top-k patch 数量
    :return: top-k patch 的索引和特征
    """
    B, C, H_fg, W_fg = foreground.shape
    _, _, H_bg, W_bg = background.shape
    ph, pw = (H_fg, W_fg)

    # 提取背景的所有 patches
    background_patches = extract_patches(background, patch_size=(ph, pw))  # (B, C*ph*pw, num_patches)
    num_patches = background_patches.shape[-1]

    # 展平前景特征
    foreground_flat = foreground.reshape(B, C, -1).transpose(1, 2)  # (B, H_fg*W_fg, C)

    # 展平背景 patches
    background_patches_flat = background_patches.transpose(1, 2)  # (B, num_patches, C*ph*pw)

    # 计算前景与背景 patches 的欧氏距离
    distances = torch.cdist(foreground_flat, background_patches_flat, p=2)  # (B, H_fg*W_fg, num_patches)

    # 找到每个前景位置对应的 top-k 最相似背景 patch
    knn_indices = distances.topk(k, dim=-1, largest=False).indices  # (B, H_fg*W_fg, k)

    # 提取 top-k 的背景 patch 特征
    topk_patches = torch.gather(
        background_patches_flat, 1, knn_indices.unsqueeze(-1).expand(-1, -1, -1, background_patches_flat.shape[-1])
    )  # (B, H_fg*W_fg, k, C*ph*pw)

    return knn_indices, topk_patches




class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class CSA(nn.Module):
    def __init__(self, in_dim, head=4):
        super().__init__()
        self.head = head
        self.scale = (in_dim//head) ** -0.5
        
        # 特征变换层
        self.to_q = nn.Linear(in_dim, in_dim)  # 前景作为Query
        self.to_kv = nn.Linear(in_dim, 2*in_dim) # 背景作为Key/Value

    def forward(self, F, M):
        """
        F: 特征图 [B,C,H,W]
        M: 前景mask [B,1,H,W] (1为前景)
        """
        B, C, H, W = F.shape
        F_flat = F.view(B, C, H*W).permute(0,2,1)  # [B, N, C]
        
        # 分割前景/背景
        fg_idx = M.view(B, H*W).nonzero(as_tuple=True)  # 前景坐标
        bg_mask = (1 - M).view(B, H*W).bool()           # 背景掩码
        
        # 生成Q/K/V
        Q = self.to_q(F_flat[fg_idx])                   # [B*N_fg, C]
        KV = self.to_kv(F_flat)                         # [B, N, 2C]
        K, V = KV.chunk(2, dim=-1)                      # [B, N, C] each
        
        # 多头处理
        Q = Q.view(B, -1, self.head, C//self.head).permute(0,2,1,3)  # [B, h, N_fg, c]
        K = K.view(B, -1, self.head, C//self.head).permute(0,2,3,1)  # [B, h, c, N]
        V = V.view(B, -1, self.head, C//self.head).permute(0,2,1,3)  # [B, h, N, c]

        # 相似度计算（仅背景区域）
        attn = (Q @ K) * self.scale                     # [B, h, N_fg, N]
        attn = attn.masked_fill(~bg_mask.unsqueeze(1).unsqueeze(2), -1e9) 
        attn = attn.softmax(dim=-1)                      # 背景区域的注意力分布
        
        # 特征聚合
        out = (attn @ V).permute(0,2,1,3).reshape(B, -1, C)  # [B, N_fg, C]
        
        # 更新前景特征
        F_new = F_flat.clone()
        F_new[fg_idx] += out.view(-1, C)  # 残差连接
        return F_new.view(B, C, H, W)


class ContrastiveCrossAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8, temperature=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = temperature
        self.scale = (in_dim // num_heads) ** -0.5
        
        # 特征转换层
        self.fg_proj = nn.Linear(in_dim, in_dim)  # 生成Q,K,V
        self.bg_proj = nn.Linear(in_dim, in_dim * 2)  # 生成K,V
        self.out_proj = nn.Linear(in_dim, in_dim)
        
        # 相似性对比参数
        self.sim_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.sim_gamma = nn.Parameter(torch.ones(1) * 0.01)  # 更小的初始值
        self.sim_gamma._clamp_min = 0.0  # 可选：限制最小值为0
        
    def forward(self, x, mask):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0,2,1)
        
        fg_mask = (mask > 0.5).float()
        bg_mask = 1 - fg_mask
        if bg_mask.sum() == 0:
            return x
        fg_mask_flat = fg_mask.flatten(2).permute(0,2,1)  # [B, HW, 1]
        bg_mask_flat = bg_mask.flatten(2).permute(0,2,1)  # [B, HW, 1]
        # 特征投影
        x_fg = x_flat * fg_mask_flat   # 前景位置保留，背景置零
        x_bg = x_flat * bg_mask_flat   # 背景位置保留，前景置零
        fg_feats = self.fg_proj(x_fg).reshape(B, -1, 1 * C)
        fg_q = torch.chunk(fg_feats, 1, dim=-1)[0]
        bg_feats = self.bg_proj(x_bg).reshape(B, -1, 2 * C)
        bg_k, bg_v = torch.chunk(bg_feats, 2, dim=-1)
        
        # 多头拆分
        fg_q = self._reshape_heads(fg_q)
        bg_k = self._reshape_heads(bg_k)
        bg_v = self._reshape_heads(bg_v)
        
        # 余弦相似度计算（归一化）
        fg_q = F.normalize(fg_q, dim=-1, eps=1e-6)
        bg_k = F.normalize(bg_k, dim=-1, eps=1e-6)
        attn = torch.matmul(fg_q, bg_k.transpose(-2,-1)) * self.scale
        
        # 温度缩放与数值裁剪
        attn = attn / self.temperature
        attn = torch.clamp(attn, min=-50.0, max=50.0)  # 防止溢出
        #self.show_mean_attention(attn, H, W)
        # 空间相似性引导（限制gamma范围）
        spatial_sim = self._calc_spatial_similarity_map(x, fg_mask)
        sim_gamma = torch.clamp(self.sim_gamma, 0.0, 1.0)  # 限制gamma范围
        attn = attn + spatial_sim * sim_gamma
        
        # 标准化
        attn = attn - attn.max(dim=-1, keepdim=True)[0]
        bg_mask_flat = bg_mask.flatten(1).unsqueeze(1).unsqueeze(-1)
        attn = attn.masked_fill(bg_mask_flat == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        # attn_heatmap = attn.mean(dim=1).cpu().numpy()  # 多头平均
        # plt.imshow(attn_heatmap[0], cmap='jet')
        # 特征聚合
        out = torch.matmul(attn, bg_v)
        out = self._reshape_back(out)
        out = self.out_proj(out) * 0.1 + x_flat  # 残差连接
        
        return out.permute(0,2,1).reshape(B, C, H, W)
    
    def _reshape_heads(self, x):
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    
    def _reshape_back(self, x):
        B, heads, N, C = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, N, -1)

    def _calc_spatial_similarity_map(self, x, fg_mask):
        """
        输出：spatial_sim [B, 1, HW, HW] - 每个 query 对所有 key 的引导 bias
        忽略前景区域的 attention bias
        """
        B, C, H, W = x.shape
        HW = H * W

        x_flat = x.view(B, C, HW)                  # [B, C, HW]
        x_norm = F.normalize(x_flat, dim=1)        # [B, C, HW]

        # 构造前景 mask，去除前景→前景区域的影响
        fg_mask_flat = fg_mask.view(B, 1, HW)
        bg_mask_flat = 1 - fg_mask_flat
        fg_self_mask = fg_mask_flat.transpose(1, 2) * fg_mask_flat  # [B, HW, HW]

        # # 前景 prototype（均值）[B, C]
        # fg_area = fg_mask.sum(dim=(2,3)) + 1e-6
        # fg_proto = (x * fg_mask).sum(dim=(2,3)) / fg_area  # [B, C]
        # fg_proto = F.normalize(fg_proto, dim=1, eps=1e-6).unsqueeze(-1)  # [B, C, 1]

        # # 每个位置和前景的相似度 [B, 1, HW]
        # sim_vec = torch.bmm(fg_proto.transpose(1,2), x_norm)  # [B, 1, HW]
        # 提取前景特征
        x_fg = x_norm * fg_mask_flat  # [B, C, HW] 仅保留前景区域

        # 提取背景特征
        x_bg = x_norm * bg_mask_flat  # [B, C, HW] 仅保留背景区域

        # # 计算局部前景 Attention
        # attn_fg = torch.bmm(x_fg.transpose(1,2), x_fg)  # [B, HW, HW]
        # attn_fg = attn_fg.masked_fill(fg_mask_flat.squeeze(1) == 0, float('-inf'))  # mask掉背景
        # attn_fg = F.softmax(attn_fg, dim=-1)

        # # 生成 refined 前景 prototype
        # refined_fg = torch.bmm(x_fg, attn_fg)        # [B, C, HW]
        # refined_fg = refined_fg * fg_mask_flat       # 只在前景位置有效

        # # 归一化 refined prototype
        # refined_fg = F.normalize(refined_fg, dim=1, eps=1e-6)

        # 每个 query 和 refined_fg 做余弦相似
        sim_vec = torch.bmm(x_bg.transpose(1, 2), x_fg)  # [B, HW, HW] 计算背景和前景之间的相似度
        sim_vec = sim_vec.clamp(min=-1, max=1)
        

        

        # 将前景区域 attention bias 置 0（或 -inf）
        #sim_bias = torch.bmm(sim_vec.transpose(1, 2), sim_vec)  # [B, HW, HW]
        sim_bias = sim_vec.masked_fill(fg_mask_flat.bool(), 0.0)  # 或 float('-inf')
        sim_bias = sim_bias.clamp(min=-5.0, max=5.0)
        return sim_bias.unsqueeze(1)

class CrossAttention1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context):
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, v)
        return self.out(output)

class CrossAttention2(nn.Module):
    def __init__(self, dims_in):
        super(CrossAttention2, self).__init__()
        self.linear = nn.Conv1d(dims_in * 2, dims_in, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        

    def forward(self, x, mask):
        # 分离前景和背景特征
        foreground_feat = x * mask  # 根据掩码提取前景特征
        background_feat = x * (1 - mask)  # 根据掩码提取背景特征

        # 展平特征图为序列
        N, C, H, W = foreground_feat.shape
        Q = foreground_feat.flatten(2).permute(0, 2, 1)  # (N, H_f*W_f, C)
        K = background_feat.flatten(2).permute(0, 2, 1)  # (N, H_b*W_b, C)
        V = K.clone()  # 通常K和V相同

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (N, H_f*W_f, H_b*W_b)
        attention_weights = self.softmax(attention_scores)

        # 计算注意力输出
        attention_output = torch.matmul(attention_weights, V)  # (N, H_f*W_f, C)

        # 调整形状以适应卷积操作
        attention_output = attention_output.permute(0, 2, 1)  # (N, C, H_f*W_f)
        foreground_feat_flat = foreground_feat.flatten(2)  # (N, C, H_f*W_f)

        # 拼接前景特征和注意力输出
        combined_features = torch.cat([foreground_feat_flat, attention_output], dim=1)

        # 通过线性层融合特征
        output_foreground_flat = self.linear(combined_features)

        # 恢复前景特征的原始形状
        output_foreground = output_foreground_flat.reshape(N, C, H, W)

        # 仅将改变应用于前景部分，背景部分保持不变
        output = output_foreground * mask + background_feat

        return output


class CrossAttention(nn.Module):
    def __init__(self, dims_in):
        super(CrossAttention, self).__init__()
        self.linear = nn.Conv1d(dims_in*2, dims_in, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim = -1)
        self.k = 1
    def forward(self, x, mask):
        mask = F.interpolate(mask.detach(), size=x.size()[2:], mode='nearest')
        B, C, H, W = x.shape
        
        flattened_features_query = x * mask
        flattened_features_support = x * (1-mask)
        
        flattened_features_query = flattened_features_query.view(B, C, -1)
        flattened_features_support = flattened_features_support.view(B, C, -1)
        
        masked_features = []
        for i in range(B):

            query_features = flattened_features_query[i].view(C, -1)
            support_features = flattened_features_support[i].view(C, -1)
            
            query = query_features.unsqueeze(0)
            support = support_features.unsqueeze(0)

            simi_matrix = torch.matmul(query.permute(0, 2, 1), support)
            
            weights_value, index = torch.topk(simi_matrix, dim=2, k=self.k)
            
            views = [support.shape[0]] + [1 if i != 2 else -1 for i in range(1, len(support.shape))]
            expanse = list(support.shape)
            expanse[0] = -1
            expanse[2] = -1
            index = index.view(views)
            index = index.expand(expanse)   
            weights_value = weights_value.view(views)
            weights_value = weights_value.expand(expanse)
            select_value = torch.gather(support, 2, index)
 
            select_value = select_value.view(1, C, self.k, -1)
            weights_value = weights_value.view(1, C, self.k, -1)
            weights_value = self.softmax(weights_value)
            fuse_tensor = weights_value * select_value
            fuse_tensor = torch.sum(fuse_tensor, -2)
            
            hybrid_feat = torch.cat((fuse_tensor, query), 1)
            hybrid_feat = self.linear(hybrid_feat)
            masked_features.append(hybrid_feat)

        refined_feat = torch.cat(masked_features, 0)
        refined_feat = refined_feat.view(B, C, H, W)
        
        return refined_feat * mask + x * (1-mask)

class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        #self.cross_attention = CrossAttention(c)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        #print(2)
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x).contiguous()
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time
    
class NAFBlock_att(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.cross_attention = ContrastiveCrossAttention(c)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        #print(1)
        inp, time, mask = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x).contiguous()
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        #修改2
        # print(x.shape)
        # print(mask.shape)
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        x = self.cross_attention(x, mask)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time

class NAFBlock_att1(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.cross_attention = CrossAttention(c)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        #print(1)
        inp, time, mask = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x).contiguous()
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        #修改2
        # print(x.shape)
        # print(mask.shape)
        mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        x = self.cross_attention(x, mask)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time



class ConditionalNAFNet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        # self.intro = nn.Conv2d(in_channels=img_channel*2+1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        self.intro = nn.Conv2d(in_channels=img_channel*2+1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        #print(chan // (2 ** (1)))
        self.cross_attns = nn.ModuleList([CrossAttention1(chan * (2 ** (i))) for i in range(len(enc_blk_nums))])
        # for num in enc_blk_nums:
        #     self.encoders.append(
        #         nn.Sequential(
        #             *[NAFBlock(chan, time_dim) for _ in range(num)]
        #         )
        #     )
        #     self.downs.append(
        #         nn.Conv2d(chan, 2*chan, 2, 2)
        #     )
        #     chan = chan * 2
        for i, num in enumerate(enc_blk_nums):
            
            if i<2:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock_att(chan, time_dim) for _ in range(middle_blk_num)]
            )

        # for num in dec_blk_nums:
        #     self.ups.append(
        #         nn.Sequential(
        #             nn.Conv2d(chan, chan * 2, 1, bias=False),
        #             nn.PixelShuffle(2)
        #         )
        #     )
        #     chan = chan // 2
        #     self.decoders.append(
        #         nn.Sequential(
        #             *[NAFBlock(chan, time_dim) for _ in range(num)]
        #         )
        #     )

        i=0
        for num in dec_blk_nums:
            #print(i)
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if i<2:
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            else:
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            i+=1

        self.padder_size = 2 ** (len(self.encoders))
        

    def forward(self, inp, cond, time, mask, reconstruction_encs=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        #x = torch.cat([x, cond, mask], dim=1)
        x = torch.cat([x, cond, mask], dim=1)
        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)
        
        encs = [x]
        i = 1
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            #x, _ = encoder([x, t])
            if i<2:
                x, _ = encoder([x, t])
            else:
                x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)
        target_size = (x.size(2), x.size(3))  # 目标大小为 (4, 4)

        # 使用双线性插值进行下采样
        #mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        x, _ = self.middle_blks([x, t, mask])
        #x, _ = self.middle_blks([x, t])
        decs = []
        i=0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            #x, _ = decoder([x, t])
            if i<2:
                x, _ = decoder([x, t])
            else:
                x, _ = decoder([x, t])
            i+=1
            decs.append(x)

        x = self.ending(x + encs[0])

        x = x[..., :H, :W]

        return x, encs, decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class ConditionalNAFNet_test(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        # self.intro = nn.Conv2d(in_channels=img_channel*2+1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
        #                       bias=True)
        self.intro = nn.Conv2d(in_channels=img_channel*2+1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        #print(chan // (2 ** (1)))
        self.cross_attns = nn.ModuleList([CrossAttention1(chan * (2 ** (i))) for i in range(len(enc_blk_nums))])
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2
        # for i, num in enumerate(enc_blk_nums):
            
        #     if i<2:
        #         self.encoders.append(
        #             nn.Sequential(
        #                 *[NAFBlock(chan, time_dim) for _ in range(num)]
        #             )
        #         )
        #     else:
        #         self.encoders.append(
        #             nn.Sequential(
        #                 *[NAFBlock_att(chan, time_dim) for _ in range(num)]
        #             )
        #         )
        #     self.downs.append(
        #         nn.Conv2d(chan, 2*chan, 2, 2)
        #     )
        #     chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock_att(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        # i=0
        # for num in dec_blk_nums:
        #     #print(i)
        #     self.ups.append(
        #         nn.Sequential(
        #             nn.Conv2d(chan, chan * 2, 1, bias=False),
        #             nn.PixelShuffle(2)
        #         )
        #     )
        #     chan = chan // 2
        #     if i<2:
        #         self.decoders.append(
        #             nn.Sequential(
        #                 *[NAFBlock_att(chan, time_dim) for _ in range(num)]
        #             )
        #         )
        #     else:
        #         self.decoders.append(
        #             nn.Sequential(
        #                 *[NAFBlock(chan, time_dim) for _ in range(num)]
        #             )
        #         )
        #     i+=1

        self.padder_size = 2 ** (len(self.encoders))
        

    def forward(self, inp, cond, time, mask, reconstruction_encs=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        #x = torch.cat([x, cond, mask], dim=1)
        x = torch.cat([x, cond, mask], dim=1)
        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)
        
        encs = [x]
        i = 1
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            x, _ = encoder([x, t])
            # if i<2:
            #     x, _ = encoder([x, t])
            # else:
            #     [x, _, _] = encoder([x, t, mask])
            encs.append(x)
            x = down(x)
        target_size = (x.size(2), x.size(3))  # 目标大小为 (4, 4)

        # 使用双线性插值进行下采样
        #mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        x, _ = self.middle_blks([x, t, mask])
        #x, _ = self.middle_blks([x, t])
        decs = []
        i=0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])
            # if i<2:
            #     [x, _, _] = decoder([x, t, mask])
            # else:
            #     x, _ = decoder([x, t])
            i+=1
            decs.append(x)

        x = self.ending(x + encs[0])

        x = x[..., :H, :W]

        return x, encs, decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
class ConditionalNAFNet_recons_1(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        #self.cross_attns = nn.ModuleList([CrossAttention1(chan * (2 ** (i))) for i in range(len(enc_blk_nums))])
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        #self.cross_attns = nn.ModuleList([CrossAttention1(chan * (2 ** (i))) for i in range(len(enc_blk_nums))])
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock_att(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** (len(self.encoders))

    def forward(self, inp, cond, time, mask):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        x = torch.cat([x, cond], dim=1)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)
        
        encs = [x]
        i = 1
        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)
        target_size = (x.size(2), x.size(3))  # 目标大小为 (4, 4)

        # 使用双线性插值进行下采样
        #mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        x, _ = self.middle_blks([x, t, mask])
        decs = []
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])
            decs.append(x)

        x = self.ending(x + encs[0])

        x = x[..., :H, :W]

        return x, encs, decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class ConditionalNAFNet_recons(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], upscale=1):
        super().__init__()
        self.upscale = upscale
        fourier_dim = width
        sinu_pos_emb = SinusoidalPosEmb(fourier_dim)
        time_dim = width * 4

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.intro = nn.Conv2d(in_channels=img_channel*2+1, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        #print(chan // (2 ** (1)))
        self.cross_attns = nn.ModuleList([CrossAttention1(chan * (2 ** (i))) for i in range(len(enc_blk_nums))])
        
        for i, num in enumerate(enc_blk_nums):
            
            if i<2:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            else:
                self.encoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock_att(chan, time_dim) for _ in range(middle_blk_num)]
            )
        i=0
        for num in dec_blk_nums:
            #print(i)
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            if i<2:
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            else:
                self.decoders.append(
                    nn.Sequential(
                        *[NAFBlock(chan, time_dim) for _ in range(num)]
                    )
                )
            i+=1

        self.padder_size = 2 ** (len(self.encoders))
        

    def forward(self, inp, cond, time, mask, reconstruction_encs=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(inp.device)

        x = inp - cond
        x = torch.cat([x, cond, mask], dim=1)

        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)
        
        encs = [x]
        
        for i, (encoder, down) in enumerate(zip(self.encoders, self.downs)):
            #print(i)
            if i<2:
                x, _ = encoder([x, t])
            else:
                x, _ = encoder([x, t])
            # if reconstruction_encs is not None:
            #     # 进行交叉注意力
            #     random_num = torch.rand(1).item()
            #     if random_num < 0.7:
            #         # 50% 的概率跳过该模块
            #         pass
            #     else:
            #         b, c, h, w = x.shape
            #         x_flat = x.view(b, c, -1).transpose(1, 2)
            #         context_flat = reconstruction_encs[i].view(b, c, -1).transpose(1, 2)
            #         #print(x_flat.size(), context_flat.size(),i)
            #         #assert x_flat.size(-1) == self.cross_attns[i].q.in_features
            #         attn_output = self.cross_attns[i](x_flat, context_flat)
            #         x = attn_output.transpose(1, 2).view(b, c, h, w)
            #print(i)
            encs.append(x)
            x = down(x)
        #print(1)
        target_size = (x.size(2), x.size(3))  # 目标大小为 (4, 4)

        # 使用双线性插值进行下采样
        #mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
        #x, _ = self.middle_blks([x, t, mask])
        x, _ = self.middle_blks([x, t, mask])
        decs = []
        i=0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            if i<2:
                x, _ = decoder([x, t])
            else:
                x, _ = decoder([x, t])
            i+=1
            decs.append(x)

        x = self.ending(x + encs[0])

        x = x[..., :H, :W]

        return x, encs, decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class CNAFNetLocal(Local_Base, ConditionalNAFNet):
    def __init__(self, *args, train_size=(1, 3, 128, 128), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ConditionalNAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


