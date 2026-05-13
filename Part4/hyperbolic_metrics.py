# hyperbolic_metrics.py
import numpy as np

def _safe_scale(x, scale=0.95):
    """将L2归一化向量缩放至双曲球内部，避免边界奇点"""
    return x * scale

def poincare_similarity(u, v, c=1.0):
    """
    计算 Poincaré 球模型下的双曲相似度
    c: 曲率参数 (c>0)。c≈0 逼近欧氏空间，c=1 强层级表征
    """
    u, v = _safe_scale(u), _safe_scale(v)
    sq_norm_u = np.sum(u**2, axis=1, keepdims=True)
    sq_norm_v = np.sum(v**2, axis=1, keepdims=True)
    sq_dist_uv = np.sum((u - v)**2, axis=1, keepdims=True)
    
    c_vec = np.full_like(sq_norm_u, c)
    denom = (1 - c_vec * sq_norm_u) * (1 - c_vec * sq_norm_v)
    denom = np.clip(denom, 1e-12, None)
    
    arg = 1 + 2 * sq_dist_uv / denom
    arg = np.clip(arg, 1.0, None)  # arccosh 定义域 >=1
    
    dist = np.arccosh(arg) / np.sqrt(c_vec)
    # 指数衰减映射为 [0,1] 相似度
    return np.exp(-2.0 * dist).flatten()

def mixture_of_curvature_similarity(pred_emb, true_emb, seq_lengths=None):
    """
    [论文1 Helm MoCE 启发] 动态曲率路由
    短文本 -> 低曲率 (近欧氏) | 长文本 -> 高曲率 (强层级)
    """
    if seq_lengths is None:
        seq_lengths = np.full(len(pred_emb), 30)  # 默认启发式长度
        
    n = len(pred_emb)
    sims = np.zeros(n)
    
    # 批量分块计算提升效率
    mask_long = seq_lengths > 25
    mask_short = ~mask_long
    
    if mask_long.any():
        sims[mask_long] = poincare_similarity(pred_emb[mask_long], true_emb[mask_long], c=1.0)
    if mask_short.any():
        sims[mask_short] = poincare_similarity(pred_emb[mask_short], true_emb[mask_short], c=0.2)
        
    return sims
