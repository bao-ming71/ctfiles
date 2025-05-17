# ctfiles
dlinverse.ipynb项目基于滤波反投影(FBP)算法构建X光重建系统，通过搭建投影数据残差梯度分析网络（3层CNN架构，含3×3卷积核/32通道/ReLU激活），建立投影数据残差与图像残差的映射关系，实现自适应梯度下降优化。
single.ipynb利用U-net对单像素X光变换进行重建，加入 Conv2DTranspose + BatchNorm 提升边缘恢复能力。U-net模型架构如下所示。
def build_enhanced_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
    c1 = Conv2D(64) → BN → Conv2D(64)     # 128x128x64
    p1 = MaxPool                              ↓
    c2 = Conv2D(128) → BN → Conv2D(128)   # 64x64x128
    p2 = MaxPool                              ↓
    c3 = Conv2D(256) → BN → Conv2D(256)   # 32x32x256
    p3 = MaxPool                              ↓

    # --- Bottleneck ---
    b = Conv2D(512) → Dropout → Conv2D(512)  # 16x16x512

    # --- Decoder ---
    u3 = Conv2DTranspose(256) + concat(c3) → Conv2D(256)x2  # 32x32
    u2 = Conv2DTranspose(128) + concat(c2) → Conv2D(128)x2  # 64x64
    u1 = Conv2DTranspose(64)  + concat(c1) → Conv2D(64)x2   # 128x128

    outputs = Conv2D(1, kernel_size=1, activation='sigmoid')
