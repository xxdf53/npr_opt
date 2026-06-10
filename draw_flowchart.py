"""
Generate publication-quality NPR architecture flowchart.
Output: flowchart_npr_architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc, Wedge
import numpy as np

import matplotlib.font_manager as fm
# Use Microsoft YaHei for CJK support
font_path = 'C:/Windows/Fonts/msyh.ttc'
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['font.size'] = 9
plt.rcParams['axes.unicode_minus'] = False

# ── Colors ──────────────────────────────────────────────────────
C_INPUT   = '#E3F2FD'  # light blue
C_NPR     = '#C8E6C9'  # light green
C_SOBEL   = '#FFCDD2'  # light red
C_BACKBONE = '#E1BEE7' # light purple
C_TKP     = '#B2EBF2'  # cyan
C_FC      = '#FFF9C4'  # light yellow
C_AUG     = '#FFE0B2'  # orange
C_EDGE    = '#37474F'
C_ARROW   = '#546E7A'
C_TEXT    = '#212121'
C_INNOV   = '#FF6F00'  # innovation highlight

fig, ax = plt.subplots(1, 1, figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False, edge_color=None, linewidth=1.5):
    """Draw a rounded box with text."""
    if edge_color is None:
        edge_color = C_EDGE
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.15,rounding_size=0.25",
                         facecolor=color, edgecolor=edge_color,
                         linewidth=linewidth, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, color=C_TEXT, zorder=3)
    return box

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.2, style='->', zorder=1):
    """Draw an arrow."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle='arc3,rad=0'),
                zorder=zorder)

def draw_label(ax, x, y, text, fontsize=7.5, color=C_INNOV, weight='bold', ha='center'):
    """Draw a label."""
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            weight=weight, color=color, zorder=4)

# ── Layout constants ─────────────────────────────────────────────
BOX_W = 2.4
BOX_H = 0.75
GAP_Y = 0.5
LEFT_X = 1.0
CENTER_X = 7.0
RIGHT_X = 14.0

# ── Title ────────────────────────────────────────────────────────
ax.text(9, 9.6, 'Multi-Scale Edge-Guided NPR with Top-K Pooling for Deepfake Detection',
        ha='center', va='center', fontsize=13, weight='bold', color=C_EDGE)
ax.text(9, 9.15, 'Overall Architecture & Four Proposed Innovations',
        ha='center', va='center', fontsize=10, color='#546E7A')

# ══════════════════════════════════════════════════════════════════
# Row 1: Input
# ══════════════════════════════════════════════════════════════════
y_input = 8.0
draw_box(ax, LEFT_X, y_input, BOX_W, BOX_H,
         'Input Image\n[B, 3, H, W]', C_INPUT, bold=True)

# ══════════════════════════════════════════════════════════════════
# Innovation ④: Frequency Attenuation (training only, dashed)
# ══════════════════════════════════════════════════════════════════
y_aug = 7.0
draw_box(ax, LEFT_X, y_aug, BOX_W, BOX_H * 0.85,
         'Freq Attenuation ④\n2×2 AvgPool → Upsample\nLerp: α=0.3, p=30%',
         C_AUG, fontsize=7.5, edge_color='#EF6C00', linewidth=2.0)

# Innovation tag
draw_label(ax, LEFT_X + BOX_W + 0.05, y_aug + BOX_H * 0.85 - 0.12,
           '← 创新④: 频率衰减增强\n  仅训练时触发\n  (削弱 f≈0.5 → 强制多频段学习)',
           fontsize=7, color='#E65100', ha='left')

# Arrow: input → freq aug
draw_arrow(ax, LEFT_X + BOX_W/2, y_input, LEFT_X + BOX_W/2, y_aug + BOX_H * 0.85 + 0.05)

# Side note: 70% skip
ax.annotate('70% skip →', xy=(LEFT_X + BOX_W + 0.6, y_input - 0.05),
            xytext=(LEFT_X + BOX_W + 1.5, y_input + BOX_H/2 + 0.2),
            arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.0,
                            connectionstyle='arc3,rad=0.3'),
            fontsize=7, color='#78909C', zorder=1)

# ══════════════════════════════════════════════════════════════════
# Innovation ①: Multi-scale NPR
# ══════════════════════════════════════════════════════════════════
y_npr = 5.5
# Main NPR block
draw_box(ax, LEFT_X, y_npr, BOX_W, BOX_H * 1.1,
         'Multi-Scale NPR ①\n├ NPR(0.25) = x - interp(x,0.25)\n├ NPR(0.50) = x - interp(x,0.50)\n└ NPR(0.75) = x - interp(x,0.75)\n→ Concat → [B, 9, H, W]',
         C_NPR, fontsize=7.2, edge_color='#2E7D32', linewidth=2.0)

draw_label(ax, LEFT_X + BOX_W + 0.05, y_npr + BOX_H * 1.1 - 0.15,
           '← 创新①: 多尺度 NPR 提取\n  覆盖 f≈0.25/0.50/0.75\n  JPEG高频压制 → 中频存活',
           fontsize=7, color='#2E7D32', ha='left')

# Arrow: freq aug → npr
draw_arrow(ax, LEFT_X + BOX_W/2, y_aug, LEFT_X + BOX_W/2, y_npr + BOX_H * 1.1 + 0.05)

# ══════════════════════════════════════════════════════════════════
# Innovation ②: Sobel Edge Guidance
# ══════════════════════════════════════════════════════════════════
y_sobel = 4.1
draw_box(ax, LEFT_X, y_sobel, BOX_W, BOX_H * 1.1,
         'Sobel Edge Guidance ②\nGray → Gx, Gy → √(Gx²+Gy²)\n→ σ(edge_norm × 5)\n→ NPR ⊙ expand_as(weight)',
         C_SOBEL, fontsize=7.2, edge_color='#C62828', linewidth=2.0)

draw_label(ax, LEFT_X + BOX_W + 0.05, y_sobel + BOX_H * 1.1 - 0.15,
           '← 创新②: Sobel 边缘引导\n  聚焦高对比度边缘区域\n  平坦区权重 → 0',
           fontsize=7, color='#C62828', ha='left')

# Arrow: npr → sobel
draw_arrow(ax, LEFT_X + BOX_W/2, y_npr, LEFT_X + BOX_W/2, y_sobel + BOX_H * 1.1 + 0.05)

# ══════════════════════════════════════════════════════════════════
# Backbone: ResNet-50
# ══════════════════════════════════════════════════════════════════
y_backbone = 2.0
draw_box(ax, CENTER_X, y_backbone, BOX_W * 1.3, BOX_H * 1.3,
         'ResNet-50 Backbone\n━━━━━━━━━━━━━━\nconv1: 9→64, k3, s2\nBatchNorm + ReLU\nMaxPool k3, s2\nlayer1: 3×Bottleneck(256)\nlayer2: 4×Bottleneck(512)\n→ [B, 512, H/8, W/8]',
         C_BACKBONE, fontsize=7)

# Arrow: sobel → backbone (turn right)
y_mid = y_sobel + BOX_H * 1.1 / 2
draw_arrow(ax, LEFT_X + BOX_W, y_mid, CENTER_X, y_mid, color=C_ARROW, lw=1.3)

# ══════════════════════════════════════════════════════════════════
# Innovation ③: Top-K Pooling
# ══════════════════════════════════════════════════════════════════
y_tkp = 0.4
# Main TKP box
draw_box(ax, CENTER_X, y_tkp, BOX_W * 1.3, BOX_H * 1.4,
         'Top-K Pooling ③\n━━━━━━━━━━━━━━━━━━━\nFlatten → [B, 512, N]\nTopK(K=5) → RBLD → vec [B, 512×5]\nRKS(K=5) → sort → vec_aux [B, 512×5]\n(eval: vec_aux = 0)',
         C_TKP, fontsize=7, edge_color='#00838F', linewidth=2.0)

draw_label(ax, CENTER_X + BOX_W * 1.3 + 0.05, y_tkp + BOX_H * 1.4 - 0.4,
           '← 创新③: Top-K Pooling\n  替代 GAP，保留稀疏强信号\n  RBLD + RKS 正则化',
           fontsize=7, color='#00838F', ha='left')

# Arrow: backbone → tkp
draw_arrow(ax, CENTER_X + BOX_W * 1.3 / 2, y_backbone,
           CENTER_X + BOX_W * 1.3 / 2, y_tkp + BOX_H * 1.4 + 0.05)

# ══════════════════════════════════════════════════════════════════
# FC + Output
# ══════════════════════════════════════════════════════════════════
y_fc = 4.5
draw_box(ax, RIGHT_X, y_fc + 0.5, BOX_W * 0.5, BOX_H * 0.6,
         'Linear\n512×K→1', C_FC, fontsize=7.5)

draw_box(ax, RIGHT_X, y_fc - 0.5, BOX_W * 0.5, BOX_H * 0.5,
         'Sigmoid\nReal / Fake', C_FC, fontsize=7.5, bold=True)

# Arrow: tkp → fc
draw_arrow(ax, CENTER_X + BOX_W * 1.3, y_tkp + BOX_H * 1.4 / 2,
           RIGHT_X + BOX_W * 0.25, y_fc + 0.5 + BOX_H * 0.6,
           color=C_ARROW, lw=1.2)

# Arrow: fc → sigmoid
draw_arrow(ax, RIGHT_X + BOX_W * 0.25, y_fc + 0.5,
           RIGHT_X + BOX_W * 0.25, y_fc - 0.5 + BOX_H * 0.5 + 0.05)

# Aux loss from TKP
draw_box(ax, CENTER_X + BOX_W * 1.3 - 0.8, y_tkp - 0.5, 1.6, 0.35,
         'BCE(FC(vec_aux), y) × 0.1', C_AUG, fontsize=6.5,
         edge_color='#EF6C00', linewidth=1.0)
draw_label(ax, CENTER_X + BOX_W * 1.3, y_tkp - 0.32, 'aux', fontsize=6,
           color='#E65100', ha='left')

# ══════════════════════════════════════════════════════════════════
# Legend: Innovation tags
# ══════════════════════════════════════════════════════════════════
legend_y = 0.2
innovs = [
    (LEFT_X, '① Multi-Scale NPR', C_NPR, '#2E7D32'),
    (LEFT_X + 4.3, '② Sobel Edge', C_SOBEL, '#C62828'),
    (CENTER_X + 3.0, '③ Top-K Pooling', C_TKP, '#00838F'),
    (LEFT_X + 4.3 + 4.3, '④ Freq Augment', C_AUG, '#EF6C00'),
]
for x, label, fc, ec in innovs:
    b = FancyBboxPatch((x, legend_y), 2.4, 0.3,
                       boxstyle="round,pad=0.08,rounding_size=0.12",
                       facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(b)
    ax.text(x + 1.2, legend_y + 0.15, label, ha='center', va='center',
            fontsize=6.5, weight='bold', color=C_TEXT)

# ══════════════════════════════════════════════════════════════════
# Right sidebar: Synergy notes
# ══════════════════════════════════════════════════════════════════
synergy_x = 15.5
synergy_y = 7.5
ax.text(synergy_x, synergy_y, '协同关系', fontsize=9, weight='bold', color=C_EDGE)
synergies = [
    '①+②: 多频段 × 空间聚焦\n   边缘伪影检测最大化',
    '①+④: 削弱 f≈0.5 → 迫使\n   使用中频分支',
    '②+③: 边缘加权 → TKP\n   几乎必然落在边缘区',
    '③+④: 更强衰减 → 更稀\n   疏信号 → TKP 价值更大',
    '①+③: 多频段特征 → 每\n   通道更多样化的 Top-K',
]
for i, s in enumerate(synergies):
    ax.text(synergy_x, synergy_y - 1.1 - i * 0.95, s, fontsize=6.5,
            color='#455A64', va='top')

plt.tight_layout(pad=1.5)
plt.savefig('flowchart_npr_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print('Saved: flowchart_npr_architecture.png')
