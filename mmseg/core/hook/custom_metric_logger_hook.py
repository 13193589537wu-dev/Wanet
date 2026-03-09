import os
import pandas as pd
from mmcv.runner import HOOKS, Hook
from mmseg.core.evaluation import EvalHook, DistEvalHook


@HOOKS.register_module()
class CustomMetricLoggerHook(Hook):
    """自定义钩子，在训练期间记录评估指标和损失到 CSV。"""

    def __init__(self, output_dir, interval=4000):
        self.output_dir = output_dir
        self.interval = interval
        self.metrics = ['mIoU', 'mDice', 'mFscore', 'loss']
        self.csv_path = os.path.join(output_dir, 'training_metrics.csv')
        self.metric_history = []

    def after_train_iter(self, runner):
        """每训练迭代后调用。"""
        if self.every_n_iters(runner, self.interval):
            # 主动触发验证流程（必须依赖已有的 EvalHook）
            for hook in runner.hooks:
                if isinstance(hook, (EvalHook, DistEvalHook)):
                    hook._do_evaluate(runner)  # 注意：调用的是内部接口
                    break

    def after_val_iter(self, runner):
        """验证后记录评估指标与损失。"""
        if self.every_n_iters(runner, self.interval):
            log_vars = runner.log_buffer.output
            metric_data = {'iteration': runner.iter}

            for metric in self.metrics:
                value = log_vars.get(metric, None)
                if value is None:
                    runner.logger.warning(f"[CustomMetricLoggerHook] `{metric}` not found in log_buffer.output.")
                metric_data[metric] = value

            self.metric_history.append(metric_data)

            # 保存为 CSV 文件
            os.makedirs(self.output_dir, exist_ok=True)
            df = pd.DataFrame(self.metric_history)
            df.to_csv(self.csv_path, index=False)
