import torch.distributed as dist

class LoggerClass:
    def __init__(self):
        self.rank = None
        if dist.is_initialized():
            self.rank = dist.get_rank()
    
    def log(self, message):
        if self.rank is None or self.rank == 0:
            print(message)
    
    def __call__(self, message):
        self.log(message)

# 创建单例实例
Logger = LoggerClass()