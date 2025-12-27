#!/usr/bin/env python3
"""
Simplified CNN Model Architecture for Audio Classification
"""
import torch
import torch.nn as nn


class FusedCNN(nn.Module):
    """
    CNN for GFCC+STE fusion features
    Input shape: (batch, 1, n_ceps+1, T)
    - Channel dimension is 1
    - Height dimension is n_ceps+1 (e.g., 21 for 20 GFCC + 1 STE)
    - Width dimension is T (time frames)
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x



# ============================================================
# Simple Convolutional Block
# ============================================================
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, pool_size=2):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# ============================================================
# Simplified Audio CNN - Easy to converge, still effective
# ============================================================
class AudioCNN(nn.Module):
    """
    Simplified CNN for audio classification.
    5 convolutional blocks with moderate depth.
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AudioCNN, self).__init__()
        
        # Convolutional blocks with gradual channel increase
        self.conv_blocks = nn.Sequential(
            # Block 1: 1 -> 32
            ConvBlock(1, 32),
            nn.Dropout2d(0.1),
            
            # Block 2: 32 -> 64
            ConvBlock(32, 64),
            nn.Dropout2d(0.1),
            
            # Block 3: 64 -> 128
            ConvBlock(64, 128),
            nn.Dropout2d(0.2),
            
            # Block 4: 128 -> 256
            ConvBlock(128, 256),
            nn.Dropout2d(0.2),
            
            # Block 5: 256 -> 256 (no increase, just deeper)
            ConvBlock(256, 256),
            nn.Dropout2d(dropout_rate),
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

# ============================================================
# Residual Block - Lightweight ResNet Style
# ============================================================
class ResBlock(nn.Module):
    """
    轻量级残差块
    Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.skip(x)
        return self.act(out)


# ============================================================
# ResNet-style Audio CNN - For Ensemble
# ============================================================
class ResLiteAudioCNN(nn.Module):
    """
    轻量级残差网络用于音频分类
    使用残差连接提升表达能力，适合ensemble
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()

        c1, c2, c3, c4 = 32, 64, 128, 256

        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(ResBlock(c1, c1), ResBlock(c1, c1))
        self.stage2 = nn.Sequential(ResBlock(c1, c2, stride=2), ResBlock(c2, c2))
        self.stage3 = nn.Sequential(ResBlock(c2, c3, stride=2), ResBlock(c3, c3))
        self.stage4 = nn.Sequential(ResBlock(c3, c4, stride=2), ResBlock(c4, c4))
        self.dp = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c4, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)
        x = self.dp(x)
        return self.fc(x)


# ============================================================
# Dilated Convolutional Block - For Increased Receptive Field
# ============================================================
class DilatedConvBlock(nn.Module):
    """
    膨胀卷积块 - 增加感受野，捕获长距离依赖
    Conv (dilated) -> BN -> ReLU -> MaxPool
    """
    def __init__(self, in_ch, out_ch, dilation=(1,2), padding=(1,2), pool=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool)
    
    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))


# ============================================================
# Dilated Audio CNN - For Ensemble with Larger Receptive Field
# ============================================================
class DilatedAudioCNN(nn.Module):
    """
    使用膨胀卷积的音频分类网络
    通过膨胀卷积增加感受野，适合捕获音频中的长距离模式
    与ResLiteAudioCNN和AudioCNN形成互补，提升ensemble效果
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.b1 = ConvBlock(1, 32)          # 普通卷积 (1 -> 32)
        self.b2 = ConvBlock(32, 64)         # 普通卷积 (32 -> 64)
        self.b3 = DilatedConvBlock(64, 128, dilation=(1,2), padding=(1,2))   # 膨胀卷积
        self.b4 = DilatedConvBlock(128, 256, dilation=(1,4), padding=(1,4))  # 更大膨胀率
        self.dp = nn.Dropout2d(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.dp(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def get_fixres_finetune_params(model: AudioCNN,
                               train_last_block_bn_only: bool = True):
    """
    为 FixRes 第二阶段微调选择要训练的参数。
    
    默认行为:
      1. 冻结大部分特征提取层
      2. 只训练:
         - 最后一个 ConvBlock 的 BN 参数
         - classifier 中的所有参数
    如果 train_last_block_bn_only=False，则会训练最后一个 ConvBlock 的所有参数 + classifier。
    """
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 找到最后一个 ConvBlock
    last_block = None
    for m in model.conv_blocks.modules():
        if isinstance(m, ConvBlock):
            last_block = m

    if train_last_block_bn_only:
        # 只训练 BN（更贴近 FixRes 原始"只修正统计量"的想法）
        for p in last_block.bn.parameters():
            p.requires_grad = True
    else:
        # 训练整个最后一个卷积块，多给一点 capacity
        for p in last_block.parameters():
            p.requires_grad = True

    # classifier 全部训练
    for p in model.classifier.parameters():
        p.requires_grad = True

    # 收集需要优化的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return trainable_params
