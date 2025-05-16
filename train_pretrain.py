import os
import platform
import argparse
import time
import math
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
import tkinter as tk
from tkinter import filedialog, messagebox

from transformers import AutoTokenizer

from model.model import numlifeLM
from model.LMConfig import LMConfig
from model.dataset import PretrainDataset

from torch.nn import CrossEntropyLoss

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


from tqdm import tqdm

def train_epoch(epoch, wandb):
    model.train()
    torch.cuda.empty_cache()
    
    # 添加进度条
    pbar = tqdm(total=len(train_loader), 
                desc=f'Epoch {epoch + 1}/{args.epochs}',
                disable=ddp and dist.get_rank() != 0)  # 在分布式训练中只显示主进程的进度条
    
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 使用 non_blocking=True 加速数据传输
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 更新进度条
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            pbar.set_postfix({
                'loss': f'{loss.item() * args.accumulation_steps:.3f}',
                'lr': f'{lr:.6f}',
                'time': f'{spend_time/(step+1)*iter_per_epoch//60:.0f}min'
            })
        pbar.update()

        # 定期保存检查点
        if (step + 1) % args.save_interval == 0:
            save_checkpoint(epoch, model, optimizer, loss.item())
    
    # 每个 epoch 结束时保存
    save_checkpoint(epoch, model, optimizer, loss.item())
    pbar.close()


def init_model(lm_config):
    # 获取绝对路径
    tokenizer_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        'model',
        'numlife_tokenizer'
    ))
    
    try:
        Logger(f"正在从路径加载分词器: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,  # 强制使用本地文件
            trust_remote_code=True   # 信任本地代码
        )
        Logger("分词器加载完成")
    except Exception as e:
        Logger(f"加载分词器失败: {e}")
        raise
        
    model = numlifeLM(lm_config).to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    
    # 移除 torch.compile()
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
        
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def print_train_info(model, dataset):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Logger(f'模型总参数量：{total_params/1e6:.3f}M')
    Logger(f'训练数据量：{len(dataset)}条')
    Logger(f'预计显存占用：{total_params*4/1024/1024:.2f}MB')


def save_checkpoint(epoch, model, optimizer, loss):
    """保存模型检查点"""
    if not ddp or dist.get_rank() == 0:  # 只在主进程保存
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': lm_config,
        }
        save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, save_path)
        Logger(f"保存检查点到: {save_path}")


def select_dataset_ui():
    """通过图形界面选择数据集文件"""
    # 创建临时的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 显示文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择训练数据集文件",
        filetypes=[
            ("JSONL files", "*.jsonl"), 
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ],
        initialdir=os.path.dirname(os.path.abspath(__file__))
    )
    
    # 如果用户选择了文件
    if file_path:
        messagebox.showinfo("选择成功", f"已选择数据集：\n{file_path}")
        return file_path
    else:
        messagebox.showwarning("未选择文件", "未选择数据集，将使用默认数据集路径")
        return None


def ask_for_ui_selection():
    """询问用户是否使用UI选择数据集"""
    if not ddp or dist.get_rank() == 0:  # 只在主进程询问
        print("\n是否使用图形界面选择训练数据集? (y/n): ", end="")
        choice = input().strip().lower()
        return choice.startswith('y')
    return False

# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="numLife Pretraining")
    parser.add_argument("--out_dir", type=str, default="out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)  # 增大 batch_size
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="numlife-Pretrain")
    parser.add_argument("--num_workers", type=int, default=4)  # 增加数据加载线程数
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=4)  # 减少梯度累积步数
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dim', default=256, type=int)       
    parser.add_argument('--n_layers', default=4, type=int)    
    parser.add_argument('--n_heads', default=4, type=int)     
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--sample_ratio", type=float, default=0.01, help="使用训练数据的比例")
    parser.add_argument("--use_ui", action="store_true", default=True, help="使用图形界面选择数据集")
    parser.add_argument("--no_ui", action="store_true", help="不使用图形界面选择数据集")
    parser.add_argument("--ask_ui", action="store_true", help="询问是否使用图形界面选择数据集")
    args = parser.parse_args()

    # 处理UI选择相关的参数
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    
    # 如果指定了不使用UI，则关闭UI选择
    if args.no_ui:
        args.use_ui = False
    
    # 如果指定询问是否使用UI
    if args.ask_ui:
        use_ui = ask_for_ui_selection()
        args.use_ui = use_ui
    
    # 如果启用UI选择，且不是分布式训练或是主进程
    if args.use_ui and (not ddp or int(os.environ.get("LOCAL_RANK", 0)) == 0):
        selected_path = select_dataset_ui()
        if selected_path:
            args.data_path = selected_path
            Logger(f"通过UI选择的数据集路径: {args.data_path}")
        else:
            # 如果使用了UI但没有选择文件，则报错并退出
            # 不再回退到任何默认路径
            Logger("错误：当使用UI选择数据集时，必须提供一个有效的数据集文件。操作已取消或未选择文件。", error=True)
            import sys
            sys.exit("错误：未通过UI选择数据集文件，训练中止。")
    else:
        # 如果不使用UI，则记录将使用命令行提供或argparse默认的数据集路径
        # 这里假设 args.data_path 已经被 argparse 设置了（或者通过其他方式）
        if not args.data_path or not os.path.isfile(args.data_path):
            Logger(f"错误：命令行提供或默认的数据集路径 '{args.data_path}' 无效或不是一个文件。", error=True)
            import sys
            sys.exit(f"错误：数据集路径 '{args.data_path}' 无效。")
        Logger(f"将使用命令行或默认的数据集路径: {args.data_path}")

    # 在此之后，args.data_path 应该是一个有效的、用户指定（通过UI或命令行）的路径
    # 或者如果UI被取消，程序已经退出

    Logger(f"最终使用的数据集路径进行训练: {args.data_path}")

    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"numlife-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=lm_config.max_seq_len,
        sample_ratio=args.sample_ratio
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=True,  # 添加 drop_last
        shuffle=False,
        num_workers=4,   # 增加 worker 数量
        persistent_workers=True,  # 保持 worker 进程
        prefetch_factor=2,       # 预加载因子
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    print_train_info(model, train_ds)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)