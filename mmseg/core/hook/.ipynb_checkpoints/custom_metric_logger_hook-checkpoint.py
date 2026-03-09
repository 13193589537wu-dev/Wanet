import os
import pandas as pd
from mmcv.runner import HOOKS, Hook
from mmseg.core import DistEvalHook

@HOOKS.register_module()
class CustomMetricLoggerHook(Hook):
    """Custom hook to log evaluation metrics and loss to CSV during training."""
    
    def __init__(self, output_dir, interval=4000):
        self.output_dir = output_dir
        self.interval = interval
        self.metrics = ['mIoU', 'mDice', 'mFscore', 'loss']  # 记录 mIoU、mDice、mFscore 和 loss
        self.csv_path = os.path.join(output_dir, 'training_metrics.csv')
        self.metric_history = []

    def after_train_iter(self, runner):
        """Called after each training iteration."""
        if runner.iter % self.interval == 0:
            # Trigger evaluation if using DistEvalHook
            for hook in runner.hooks:
                if isinstance(hook, DistEvalHook):
                    hook.evaluate(runner)
                    break

    def after_val_iter(self, runner):
        """Called after validation to log metrics and loss."""
        if runner.iter % self.interval == 0:
            # Extract metrics and loss from runner's log_buffer
            log_vars = runner.log_buffer.output
            metric_data = {'iteration': runner.iter}
            for metric in self.metrics:
                if metric in log_vars:
                    metric_data[metric] = log_vars[metric]
                else:
                    metric_data[metric] = None
            self.metric_history.append(metric_data)
            
            # Save to CSV
            df = pd.DataFrame(self.metric_history)
            os.makedirs(self.output_dir, exist_ok=True)
            df.to_csv(self.csv_path, index=False)