"""
Bonn EEG 癫痫检测项目 - 配置文件
"""
import os

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SUBSET_FOLDERS = {
    "Z": os.path.join(DATA_DIR, "Z"),  # Set A: 正常人，睁眼
    "O": os.path.join(DATA_DIR, "O"),  # Set B: 正常人，闭眼
    "N": os.path.join(DATA_DIR, "N"),  # Set C: 癫痫患者，发作间期（对侧半球）
    "F": os.path.join(DATA_DIR, "F"),  # Set D: 癫痫患者，发作间期（癫痫灶）
    "S": os.path.join(DATA_DIR, "S"),  # Set E: 癫痫患者，发作期
}

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ============================================================
# 数据参数
# ============================================================
SAMPLE_RATE = 173.61          # Bonn 数据集采样率 (Hz)
POINTS_PER_FILE = 4097        # 每个文件中的数据点数
RECORDING_DURATION = 23.6     # 每段录音时长 (秒)

# ============================================================
# 滑动窗口分割参数
# ============================================================
WINDOW_SIZE = 1024            # 滑动窗口大小 (数据点)
WINDOW_OVERLAP = 0.5          # 重叠率 (50%)
WINDOW_STRIDE = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))  # 步长 = 512

# ============================================================
# 分类任务配置
# ============================================================
# 二分类: 正常(Z+O) vs 癫痫发作(S)
BINARY_LABELS = {
    "Z": 0, "O": 0,  # Normal
    "S": 1,           # Seizure
}
BINARY_CLASS_NAMES = ["Normal", "Seizure"]

# 三分类: 正常(Z+O) vs 发作间期(N+F) vs 癫痫发作(S)
THREE_CLASS_LABELS = {
    "Z": 0, "O": 0,  # Normal
    "N": 1, "F": 1,  # Inter-ictal
    "S": 2,           # Seizure (Ictal)
}
THREE_CLASS_NAMES = ["Normal", "Inter-ictal", "Seizure"]

# 五分类: Z vs O vs N vs F vs S
FIVE_CLASS_LABELS = {
    "Z": 0, "O": 1, "N": 2, "F": 3, "S": 4,
}
FIVE_CLASS_NAMES = ["Z (Normal EO)", "O (Normal EC)", "N (Inter-ictal)", "F (Inter-ictal)", "S (Seizure)"]

# ============================================================
# 模型超参数
# ============================================================
CNN_FILTERS = [64, 128]       # 1D-CNN 各层卷积核数量
CNN_KERNEL_SIZE = 7           # 卷积核大小
LSTM_UNITS = [128, 64]        # Bi-LSTM 各层隐藏单元数
ATTENTION_UNITS = 64          # Self-Attention 维度
DROPOUT_RATE = 0.4            # Dropout 比率
DENSE_UNITS = 64              # 全连接层单元数

# ============================================================
# 训练超参数
# ============================================================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 15      # 早停耐心值
REDUCE_LR_PATIENCE = 7        # 学习率衰减耐心值
REDUCE_LR_FACTOR = 0.5        # 学习率衰减因子

# ============================================================
# 交叉验证
# ============================================================
K_FOLDS = 10                  # K 折交叉验证
RANDOM_SEED = 42
