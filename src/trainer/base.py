from src.config import TrainerConfig
from src.logger import BaseLogger


class BaseTrainer:
    def __init__(self, cfg: TrainerConfig):
        self.cfg = cfg
        self.logger = BaseLogger(cfg.log_file_name)

    def clear_log(self):
        self.logger.clear()

    def write_log(self, *args, sep=" ", end="\n"):
        self.logger.write_log(*args, sep=sep, end=end)
