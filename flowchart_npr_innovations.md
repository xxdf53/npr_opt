# NPR 深度伪造检测优化 — 流程图

## 图1: 整体架构流程（训练 + 推理）

```mermaid
flowchart TB
    subgraph INPUT["📥 输入"]
        IMG["Input Image RGB<br/>[B, 3, H, W]"]
    end

    subgraph AUG["🔧 创新 ④: 频率衰减增强（仅训练时）"]
        direction TB
        PROB["概率: 30%"]
        DOWN["2×2 AvgPool → 抹除 f≈0.5"]
        UP["Bilinear Upsample → 恢复尺寸"]
        LERP["Lerp: (1-α)·x + α·lowpass<br/>α = 0.3"]
        PROB --> DOWN --> UP --> LERP
        NOTE_AUG["⚠️ 仅训练时，概率触发<br/>逼网络不能只依赖 f=0.5<br/>一次增强覆盖 JPEG/Blur/Resize"]
    end

    subgraph NPR["🔬 创新 ①: 多尺度 NPR 提取"]
        direction TB
        S025["Scale = 0.25<br/>f ≈ 0.25 (中频)"]
        S050["Scale = 0.50<br/>f ≈ 0.50 (高频)"]
        S075["Scale = 0.75<br/>f ≈ 0.33 + 混叠"]
        NPR_EQ["NPR = x − interpolate(x, scale)<br/>最近邻插值残差"]
        CONCAT["Concat → [B, 9, H, W]<br/>× 2/3 normalization"]
        S025 --> NPR_EQ
        S050 --> NPR_EQ
        S075 --> NPR_EQ
        NPR_EQ --> CONCAT
        NOTE_NPR["✅ 多频段覆盖：当 JPEG 破坏<br/>最高频时，中频信号仍然存活"]
    end

    subgraph SOBEL["🎯 创新 ②: Sobel 边缘引导"]
        direction TB
        GRAY["RGB → Gray [B, 1, H, W]"]
        GX["Sobel X: [-1,0,1; -2,0,2; -1,0,1]/8"]
        GY["Sobel Y: [-1,-2,-1; 0,0,0; 1,2,1]/8"]
        EDGE["Edge = √(Gx² + Gy²)"]
        NORM["Normalize: edge / (max + ε)"]
        SIG["Weight = σ(edge_norm × 5)"]
        WEIGHTED["NPR ← NPR ⊙ Weight<br/>(expand_as: 9ch 共享同一权重)"]
        GRAY --> GX
        GRAY --> GY
        GX --> EDGE
        GY --> EDGE
        EDGE --> NORM --> SIG --> WEIGHTED
        NOTE_SOBEL["✅ 生成器伪影在边缘处信号最强<br/>平坦区几乎为零 → 聚焦高对比度区域"]
    end

    subgraph BACKBONE["🧠 ResNet-50 Backbone"]
        direction TB
        CONV1["conv1: 9→64, k3, s2"]
        BN1["BatchNorm + ReLU"]
        MAXP["MaxPool k3 s2"]
        L1["layer1: 3× Bottleneck(256)<br/>H/4 × W/4"]
        L2["layer2: 4× Bottleneck(512)<br/>H/8 × W/8"]
        FEAT["Feature Maps<br/>[B, 512, H/8, W/8]"]
        CONV1 --> BN1 --> MAXP --> L1 --> L2 --> FEAT
    end

    subgraph TKP["🔝 创新 ③: Top-K Pooling"]
        direction TB
        FLAT["Flatten: [B, 512, N]"]
        TOPK["Top-K Selection<br/>每通道取最强 K=5 值"]
        RBLD["RBLD: Rank-Based<br/>Linear Dropout<br/>p = linspace(0.1→0.3,K)<br/>高排序值 → 低丢弃率"]
        VEC["主向量 vec<br/>[B, 512×5]"]
        RKS["RKS: Random-K Sampling<br/>随机采样 K 个位置 → 排序<br/>辅助梯度路径"]
        VEC_AUX["辅助向量 vec_aux<br/>[B, 512×5]<br/>(eval 时为零)"]
        FLAT --> TOPK --> RBLD --> VEC
        FLAT --> RKS --> VEC_AUX
        NOTE_TKP["✅ 稀疏强信号保留<br/>JPEG 后只有少数位置仍有可检测信号<br/>TKP 只取这些位置"]
    end

    subgraph CLASS["🏷️ 分类"]
        FC["FC: 512×K → 1"]
        LOSS["BCEWithLogitsLoss"]
        AUX_LOSS["辅助损失: α × BCE(FC(vec_aux), label)<br/>α = 0.1"]
    end

    subgraph OUTPUT["📊 输出"]
        REAL["Real (0)"]
        FAKE["Fake (1)"]
    end

    IMG --> PROB
    PROB --"70% 不触发"--> S025
    AUG --> S025
    S025 --> CONCAT
    CONCAT --> WEIGHTED
    WEIGHTED --> CONV1
    FEAT --> FLAT
    VEC --> FC
    VEC_AUX --> AUX_LOSS
    FC --> LOSS
    LOSS --> REAL
    LOSS --> FAKE
    AUX_LOSS -.-> LOSS

    style INPUT fill:#e1f5fe,stroke:#0288d1
    style AUG fill:#fff3e0,stroke:#f57c00
    style NPR fill:#e8f5e9,stroke:#388e3c
    style SOBEL fill:#fce4ec,stroke:#c62828
    style BACKBONE fill:#f3e5f5,stroke:#7b1fa2
    style TKP fill:#e0f7fa,stroke:#00838f
    style CLASS fill:#fff8e1,stroke:#f9a825
    style OUTPUT fill:#efebe9,stroke:#4e342e
```

---

## 图2: 数据流与维度变化

```mermaid
flowchart LR
    subgraph S1["阶段1: 输入"]
        A["[B, 3, H, W]<br/>RGB Image"]
    end

    subgraph S2["阶段2: 多尺度NPR"]
        B["[B, 9, H, W]<br/>3 scales × 3 RGB<br/>Multi-band residuals"]
    end

    subgraph S3["阶段3: 边缘引导"]
        C["[B, 9, H, W]<br/>× sigmoid(edge)<br/>Spatially weighted"]
    end

    subgraph S4["阶段4: 下采样"]
        D1["[B, 64, H/2, W/2]<br/>conv1 + maxpool"]
        D2["[B, 256, H/4, W/4]<br/>layer1"]
        D3["[B, 512, H/8, W/8]<br/>layer2"]
    end

    subgraph S5["阶段5: 池化"]
        E["[B, 512×K]<br/>Top-K per channel<br/>Sparse aggregation"]
    end

    subgraph S6["阶段6: 输出"]
        F["[B, 1]<br/>Real / Fake"]
    end

    A -->|"创新①"| B
    B -->|"创新②"| C
    C -->|"ResNet"| D1
    D1 --> D2
    D2 --> D3
    D3 -->|"创新③"| E
    E -->|"FC"| F

    style S2 fill:#e8f5e9,stroke:#388e3c
    style S3 fill:#fce4ec,stroke:#c62828
    style S5 fill:#e0f7fa,stroke:#00838f
```

---

## 图3: 四项创新间的协同关系

```mermaid
flowchart TB
    INNOV["🎯 核心目标<br/>提升 JPEG/压缩鲁棒性"]
    
    I1["① 多尺度 NPR<br/>频域扩展<br/>0.25 / 0.50 / 0.75"]
    I2["② Sobel 边缘引导<br/>空间聚焦<br/>sigmoid(边缘强度)"]
    I3["③ Top-K Pooling<br/>稀疏聚合<br/>K=5 + RBLD + RKS"]
    I4["④ 频率衰减增强<br/>训练正则化<br/>概率削弱 f≈0.5"]
    
    INNOV --> I1
    INNOV --> I2
    INNOV --> I3
    INNOV --> I4
    
    I1 --"互补: 多频段输入<br/>为Sobel提供更多<br/>可筛选的信号"--> I2
    I2 --"互补: 空间加权<br/>使TKP的top-k几乎<br/>必然落在边缘区"--> I3
    I4 --"协同: 削弱f≈0.5<br/>逼网络使用<br/>I1的多频段"--> I1
    I4 --"协同: 更强衰减<br/>→更稀疏信号<br/>→TKP价值更大"--> I3
    
    I1 --"⚠️ Sobel 在 RGB 上<br/>做边缘，共享给所有频段"--> I2
    I2 --"⚠️ 双重抑制风险:<br/>非边缘强信号位置<br/>Sobel+低排名被筛除"--> I3
```

---

## 创新点定位总结

| 创新 | 管线位置 | 作用维度 | 对抗退化 |
|------|---------|---------|---------|
| ① 多尺度 NPR | 输入预处理 | **频率维度** | JPEG 高频压制 |
| ② Sobel 边缘引导 | 特征加权 | **空间维度** | 平坦区噪声 |
| ③ Top-K Pooling | 聚合策略 | **聚合维度** | 信号稀释 |
| ④ 频率衰减增强 | 训练增强 | **数据维度** | 过拟合 f=0.5 |
