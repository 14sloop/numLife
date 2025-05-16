import torch
import os
from pathlib import Path
from transformers import AutoTokenizer
from model.model import numlifeLM
from model.LMConfig import LMConfig

class Inference:
    def __init__(self, model_path, tokenizer_path):
        # 确保使用绝对路径
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        model_path = Path(model_path)  # 使用传入的完整路径
        tokenizer_path = self.base_path / tokenizer_path
        
        print(f"检查文件路径:")
        print(f"分词器路径: {tokenizer_path}")
        print(f"分词器文件是否存在: {(tokenizer_path / 'tokenizer.json').exists()}")
        print(f"模型路径: {model_path}")
        print(f"模型文件是否存在: {model_path.exists()}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载配置
        self.config = LMConfig(
            dim=256,
            n_layers=4,
            max_seq_len=128,
            use_moe=False
        )
        
        # 加载分词器
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer_file = tokenizer_path / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(f"分词器文件不存在: {tokenizer_file}")
                
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=str(tokenizer_file),
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>"
            )
            print("分词器加载成功")
        except Exception as e:
            print(f"分词器加载错误: {e}")
            raise
        
        # 加载模型
        if not Path(model_path).exists():
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        self.model = numlifeLM(self.config).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"模型加载完成，使用设备: {self.device}")
    
    @torch.no_grad()
    def generate(self, prompt, max_length=50, temperature=0.6, top_p=0.9, top_k=50):
        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_len,
            padding=True
        ).to(self.device)
        
        input_ids = inputs["input_ids"]
        vocab_size = self.tokenizer.vocab_size
        
        # 生成
        output_ids = input_ids
        for _ in range(max_length):
            outputs = self.model(output_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Top-K 过滤
            top_k = min(top_k, next_token_logits.size(-1))
            values, indices = torch.topk(next_token_logits, top_k)
            mask = next_token_logits < values[:, [-1]]
            next_token_logits[mask] = float('-inf')
            
            # Top-p (nucleus) 过滤
            probs = torch.softmax(next_token_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 0] = 0  # 保留最高概率的token
            next_token_logits[mask] = float('-inf')
            
            # 采样
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # 如果生成的token超出词表范围，停止生成
            if next_token[0, 0].item() >= vocab_size:
                break
                
            output_ids = torch.cat([output_ids, next_token], dim=1)
            
            if output_ids.shape[1] > self.config.max_seq_len:
                break
        
        # 解码输出，跳过输入部分
        input_length = input_ids.shape[1]
        generated_ids = output_ids[0, input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()

if __name__ == "__main__":
    # 使用项目根目录的相对路径
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = root_dir / "out/pretrain_256.pth"
    tokenizer_path = "model/numlife_tokenizer"
    
    print(f"模型文件路径: {model_path}")
    print(f"模型文件是否存在: {model_path.exists()}")
    
    # 初始化推理器
    try:
        inferencer = Inference(str(model_path), tokenizer_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        print(f"当前工作目录: {os.getcwd()}")
        exit(1)
    
    # 交互式对话
    print("模型已准备就绪，请输入提示词(输入 'exit' 退出):")
    while True:
        prompt = input("\n用户: ")
        if prompt.lower() == 'exit':
            break
            
        response = inferencer.generate(
            prompt,
            max_length=30,        # 更短的生成长度
            temperature=0.6,      # 降低温度使输出更稳定
            top_p=0.9,           # 适中的 nucleus sampling
            top_k=50             # 添加 top-k 过滤
        )
        print(f"\n助手: {response}")