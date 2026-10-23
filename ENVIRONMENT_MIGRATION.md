# 环境迁移指南

本文档说明如何将 `deimv2` conda 环境迁移到另一个服务器。

## 📦 导出的文件

1. **`deimv2_environment.yml`** - Conda 环境配置文件（推荐使用）
2. **`deimv2_requirements.txt`** - Pip requirements 文件（备用）

## 🚀 在新服务器上重建环境

### 方法 1：使用 Conda（推荐）

```bash
# 1. 将环境文件复制到新服务器
# scp deimv2_environment.yml user@new_server:/path/to/destination/

# 2. 在新服务器上创建环境
conda env create -f deimv2_environment.yml

# 3. 激活环境
conda activate deimv2

# 4. 验证安装
python -c "import torch; print(torch.__version__)"
```

### 方法 2：使用 Pip（如果 Conda 不可用）

```bash
# 1. 创建新的 conda 环境（Python 3.11）
conda create -n deimv2 python=3.11 -y
conda activate deimv2

# 2. 安装依赖
pip install -r deimv2_requirements.txt
```

## ⚠️ 注意事项

### 1. CUDA 版本兼容性

确保新服务器的 CUDA 版本与当前环境兼容：

```bash
# 检查当前 CUDA 版本
nvcc --version

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"
```

如果 CUDA 版本不同，可能需要重新安装 PyTorch：

```bash
# 例如：CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 系统依赖

某些包可能需要系统级别的依赖，例如：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# CentOS/RHEL
sudo yum install -y gcc gcc-c++ make openssl-devel libffi-devel python3-devel
```

### 3. 验证关键包

```bash
conda activate deimv2
python -c "
import torch
import torchvision
import numpy
import cv2
print('✅ 所有关键包已安装')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
"
```

## 📋 环境信息

- **环境名称**: deimv2
- **Python 版本**: 3.11
- **导出时间**: 2024-12-22

## 🔧 故障排除

### 问题 1: Conda 环境创建失败

```bash
# 尝试使用 --no-deps 选项，然后手动安装依赖
conda env create -f deimv2_environment.yml --no-deps
conda activate deimv2
pip install -r deimv2_requirements.txt
```

### 问题 2: 包版本冲突

```bash
# 查看冲突的包
conda env create -f deimv2_environment.yml --dry-run

# 手动解决冲突后，使用 pip 安装
pip install -r deimv2_requirements.txt --no-deps
```

### 问题 3: CUDA 相关错误

```bash
# 重新安装 PyTorch（根据新服务器的 CUDA 版本）
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 迁移检查清单

- [ ] 复制 `deimv2_environment.yml` 到新服务器
- [ ] 复制 `deimv2_requirements.txt` 到新服务器（备用）
- [ ] 在新服务器上创建 conda 环境
- [ ] 验证 Python 版本
- [ ] 验证 PyTorch 和 CUDA
- [ ] 验证所有关键包
- [ ] 测试运行训练脚本

## 🎯 快速迁移命令

```bash
# 在新服务器上执行
cd /path/to/DEIMv2
conda env create -f deimv2_environment.yml
conda activate deimv2

# 验证
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

