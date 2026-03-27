# AI Assisted Epilepsy Detection using EEG Signals

基于 Bonn 大学 EEG 数据集的癫痫自动检测系统，使用 **1D-CNN + Bi-LSTM + Self-Attention** 混合深度学习架构。

## 项目结构

```
Bonn/
├── Z/              # Set A: 正常人，睁眼 (100 个文件)
├── O/              # Set B: 正常人，闭眼 (100 个文件)
├── N/              # Set C: 癫痫患者，发作间期 (100 个文件)
├── F/              # Set D: 癫痫患者，发作间期 (100 个文件)
├── S/              # Set E: 癫痫患者，发作期 (100 个文件)
├── config.py       # 全局配置参数
├── data_loader.py  # 数据加载与预处理
├── model.py        # 模型架构定义
├── train.py        # 训练与评估脚本
├── visualize.py    # 可视化 EDA 脚本
├── requirements.txt
└── README.md
```

## 环境配置

### 方式一：使用 uv (推荐)

```bash
# 创建虚拟环境
uv venv Bonn --python 3.11

# 激活虚拟环境
# Windows PowerShell:
.\Bonn\Scripts\Activate.ps1
# Windows CMD:
.\Bonn\Scripts\activate.bat
# Linux/Mac:
source Bonn/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 方式二：使用 pip

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## 运行方式

### 1. 数据探索性分析 (EDA)

```bash
python visualize.py
```

会在 `output/` 目录生成:
- `signal_comparison.png` - 五类信号波形对比
- `fft_comparison.png` - 频域分析
- `amplitude_distribution.png` - 振幅分布

### 2. 快速训练 (单次 train/test 分割)

```bash
# 二分类: Normal vs Seizure (推荐先跑这个验证环境)
python train.py --task binary --model hybrid --mode single

# 三分类: Normal vs Inter-ictal vs Seizure
python train.py --task three --model hybrid --mode single

# 五分类: Z vs O vs N vs F vs S
python train.py --task five --model hybrid --mode single
```

### 3. K-Fold 交叉验证训练 (正式实验)

```bash
# 二分类 10-Fold CV
python train.py --task binary --model hybrid --mode kfold --folds 10

# 对比实验：纯 Bi-LSTM (无 CNN)
python train.py --task binary --model bilstm --mode kfold --folds 10
```

### 4. 命令行参数说明

| 参数       | 选项                           | 说明                              |
|-----------|-------------------------------|----------------------------------|
| `--task`  | `binary`, `three`, `five`     | 分类任务                          |
| `--model` | `hybrid`, `bilstm`            | 模型架构                          |
| `--mode`  | `single`, `kfold`             | 训练模式                          |
| `--folds` | 整数 (默认 10)                 | K-Fold 折数                       |

## 在百度云开发机上运行 (SSH)

### 步骤 1: 申请百度云 GPU 开发机

1. 访问 [百度智能云 AI Studio](https://aistudio.baidu.com/) 或百度云 BCC
2. 创建一个带 GPU 的实例 (推荐 V100 或 A100)
3. 获取 SSH 连接信息 (IP、端口、用户名)

### 步骤 2: 上传项目文件

```bash
# 在本地终端执行 (将 <user>@<ip> 替换为你的服务器信息)
scp -r "C:\Users\Arctic Zvan\Desktop\Bonn" <user>@<ip>:~/Bonn
```

或者用 VS Code / Cursor 的 Remote-SSH 插件直接连接后拖拽文件。

### 步骤 3: 在服务器上配置环境

```bash
# SSH 连接到服务器
ssh <user>@<ip> -p <port>

# 进入项目目录
cd ~/Bonn

# 安装 uv (如果服务器没有)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境并安装依赖
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 步骤 4: 开始训练

```bash
# 先快速验证环境没问题
python train.py --task binary --model hybrid --mode single

# 正式训练 (GPU 加速)
python train.py --task binary --model hybrid --mode kfold --folds 10

# 后台运行 (SSH 断开后继续训练)
nohup python train.py --task binary --model hybrid --mode kfold > training.log 2>&1 &

# 查看训练日志
tail -f training.log
```

## 模型架构

```
Input (1024, 1)
    ↓
Conv1D(64) → BN → ReLU → MaxPool → Dropout
    ↓
Conv1D(128) → BN → ReLU → MaxPool → Dropout
    ↓
Bi-LSTM(128) → Dropout
    ↓
Bi-LSTM(64) → Dropout
    ↓
Self-Attention
    ↓
Dense(64) → Dropout
    ↓
Output (Sigmoid/Softmax)
```

## 预期结果

根据文献参考:
- **二分类** (Normal vs Seizure): 预期准确率 > 98%
- **三分类**: 预期准确率 > 95%
- **五分类**: 预期准确率 > 90%

## 参考文献

1. Acharya et al. (2017) - Deep CNN for seizure detection
2. Hussein et al. (2018) - Robust LSTM for EEG seizure detection
3. Thara et al. (2019) - Stacked Bi-LSTM for seizure detection (99% on Bonn)
4. Ullah et al. (2018) - Pyramidal 1D-CNN
