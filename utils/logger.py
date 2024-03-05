import os
import logging
import platform
import torch
from datetime import datetime

class Logger(object):
    def __init__(self, path):
        if os.path.isfile(path):
            os.remove(path)
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        fh = logging.FileHandler(path, mode="a")
        sh = logging.StreamHandler()
        self._logger.addHandler(fh)
        self._logger.addHandler(sh)
    
    def args(self, args):
        """logging arguments settings"""
        args_info = str(args).replace(" ", "\n  ").replace("(", "(\n  ").replace(")", "\n)")
        self._logger.info(f"{args_info}")
        return
    
    def envs(self, device):
        """logging running environment"""
        self._logger.info(f"{'python':<6} version: {platform.python_version()}")
        self._logger.info(f"{'torch':<6} version: {torch.__version__}")
        self._logger.info(f"{'cuda':<6} version: {torch.version.cuda}")
        self._logger.info(f"{'cudnn':<6} version: {torch.backends.cudnn.version()}")
        device_prop = torch.cuda.get_device_properties(device)
        self._logger.info(f"{'device':<6} version: {device_prop.name} ({device_prop.total_memory/(1024**3):.2f} GB)")
        return
    
    def hint(self, message):
        """logging hint message"""
        self._logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => {message}")
        return
    
    def mesg(self, message):
        """logging common information"""
        self._logger.info(message)
        return
    
    def eval(self, result):
        """logging pruning effect evaluation"""
        origin_top1, origin_top5, origin_flops, origin_params = tuple(result[0])
        pruned_top1, pruned_top5, pruned_flops, pruned_params = tuple(result[1])
        self._logger.info(f"{'top@1':<6}: {origin_top1:>6.2f}% -> {pruned_top1:>6.2f}% - drop: {origin_top1-pruned_top1:>5.2f}%")
        if origin_top5 is not None:
            self._logger.info(f"{'top@5':<6}: {origin_top5:>6.2f}% -> {pruned_top5:>6.2f}% - drop: {origin_top5-pruned_top5:>5.2f}%")
        base, tag = (1e6, "M") if origin_flops / 1e9 < 1 else (1e9, "G")
        self._logger.info(f"{'flops':<6}: {origin_flops/base:>6.2f}{tag} -> {pruned_flops/base:>6.2f}{tag} - drop: {(origin_flops-pruned_flops)/origin_flops*100:>5.2f}%")
        base, tag = (1e6, "M") if origin_params / 1e9 < 1 else (1e9, "G")
        self._logger.info(f"{'params':<6}: {origin_params/base:>6.2f}{tag} -> {pruned_params/base:>6.2f}{tag} - drop: {(origin_params-pruned_params)/origin_params*100:>5.2f}%")
        return