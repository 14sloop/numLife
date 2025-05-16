# Numlife (重构与增强版)

本项目基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind) 项目进行重构和功能增强，旨在提供一个更易用、功能更丰富的轻量级大语言模型训练和交互框架。特别地，**本项目致力于探索和优化，使得CPU也能在较短时间内完成大模型相关的训练和推理任务，从而降低对高端硬件的依赖。未来，本项目将致力于探索和优化，从实现更强大的功能，到降低对高端硬件的依赖，最终实现更广泛的应用。**

## 项目背景

原 MiniMind 项目为一个优秀的轻量级大语言模型实现。本项目在其坚实基础上，着重进行了代码结构优化、新功能扩展以及用户体验的提升。其中，显著的改进包括增加了图形化的聊天交互界面 (`chat_gui.py`) 和更为灵活、用户友好的训练数据集选择机制。

## 主要特性与修改

* **代码重构**: 对原有代码结构进行了梳理和模块化优化，以提高代码的可读性、可维护性和可扩展性。
* **图形化聊天界面 (Chat UI)**: 新增 `chat_gui.py` 脚本，提供了一个基于 Tkinter 的图形化聊天界面，方便用户与训练好的模型进行直观的交互和效果测试。
* **CPU友好与效率**: 进行了针对性考量和优化，旨在使得模型在CPU环境下也能在相对较短的时间内进行有效的运算和推理，方便更多用户使用。
* **增强的数据集选择**:
    * 预训练脚本 (`train_pretrain.py`) 支持通过命令行参数 `--use_ui` 或 `--ask_ui` 启动图形化文件选择器，方便用户指定训练数据集。
    * 改进了数据集路径的处理逻辑：当启用UI选择但用户未实际选择文件时，脚本会明确报错并中止运行，避免因意外使用默认或无效路径导致的问题。
* **模型与分词器**:
    * 核心模型命名为 `numlifeLM` (主要实现位于 `model/model.py`)。
    * 使用分词器 `numlife_tokenizer` (相关文件应位于 `model/numlife_tokenizer/`)。
* **模块化训练流程**:
    * `train_pretrain.py`: 用于语言模型的预训练。
    * `train_sft.py` (如果存在): 用于模型的指令微调（Supervised Fine-Tuning）。
* **灵活的配置管理**: 通过 `LMConfig` 类 (位于 `model/LMConfig.py`) 集中管理模型的各项配置参数。
* **数据集处理**: `PretrainDataset` 类 (位于 `model/dataset.py`) 负责预训练数据的加载和预处理。
* **改进的日志与进度显示**: 
    * 引入 `tqdm` 库，为训练循环添加了交互式进度条。
    * `loss_fct` (CrossEntropyLoss) 在全局范围定义。
    * 日志通过 `pbar.set_postfix` 更新到进度条。
    * 在 `train_epoch` 开始处添加了 `torch.cuda.empty_cache()` 调用 (原始版本中无此调用)。

* **分布式训练支持 (DDP)**: 保留并可能进一步优化了原有的基于 `torch.distributed.DistributedDataParallel` 的多GPU分布式训练能力。
* **W&B 集成**: 支持通过 Weights & Biases (wandb) 对实验过程进行跟踪、可视化和管理。
* **检查点保存**: 通过独立的 `save_checkpoint` 函数保存更全面的检查点，包括 `epoch` 号、完整的模型权重、优化器状态、当前损失值以及 `LMConfig`。文件名格式为 `checkpoint_epoch_{epoch}.pt`。

## 环境设置

1.  **克隆仓库** (如果项目已托管)
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  **创建并激活Python虚拟环境**
    推荐使用 `venv` 或 `conda` 创建独立的虚拟环境。
    ```bash
    # 使用 venv
    python3 -m venv venv_numlife
    # Linux/macOS:
    source venv_numlife/bin/activate
    # Windows (CMD):
    # venv_numlife\Scripts\activate.bat

    # 或者使用 conda
    # conda create -n numlife_env python=3.9  # 可指定python版本
    # conda activate numlife_env
    ```

3.  **安装依赖**
    根据您的 `train_pretrain.py` 脚本中的导入，创建一个 `requirements.txt` 文件。以下是一个基于您提供脚本的示例内容：
    ```
    torch
    transformers
    tqdm
    # pandas # 如果您的项目最终会用到pandas，请取消注释
    # wandb  # 如果希望使用W&B日志功能，请取消注释
    ```
    然后在激活的虚拟环境中运行：
    ```bash
    pip install -r requirements.txt
    ```
    *注意: `tkinter` 通常是Python标准库的一部分。如果您的Python环境不包含它 (例如某些精简的Docker镜像或特定的Linux发行版)，可能需要单独安装 (例如在Debian/Ubuntu上使用 `sudo apt-get install python3-tk`)。*

## 数据集准备

* **预训练数据**: `train_pretrain.py` 脚本期望的数据格式为 **JSONL** (`.jsonl`)，即文件中的每一行都是一个独立的JSON对象，代表一个训练样本。
* **分词器 (Tokenizer)**:
    * 确保您的分词器文件 (例如 `vocab.json`, `merges.txt`, `tokenizer_config.json` 等) 存放在 `model/numlife_tokenizer/` 目录中 (相对于 `train_pretrain.py` 脚本的位置)。
    * 脚本会使用 `transformers.AutoTokenizer.from_pretrained` 从此路径加载分词器。

## 如何运行

### 预训练 (`train_pretrain.py`)

#### 单设备 (CPU/单GPU) 训练

```bash
python train_pretrain.py [参数...]