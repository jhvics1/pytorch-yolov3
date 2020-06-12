import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # self.writer = SummaryWriter(log_dir=log_dir)
        self.writer = SummaryWriter(
            log_dir=os.path.join(log_dir, datetime.now().isoformat()))

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
