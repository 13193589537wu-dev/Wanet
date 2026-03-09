from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class EarlyStoppingHook(Hook):
    """在验证指标停止改进一段时间后提前终止训练。"""

    def __init__(self, patience=10, metric='mIoU', rule='greater', min_delta=0.0):
        self.patience = patience
        self.metric = metric
        self.rule = rule
        self.min_delta = min_delta
        self.best_score = float('-inf') if rule == 'greater' else float('inf')
        self.best_iter = 0
        self.counter = 0

    def before_run(self, runner):
        self.counter = 0
        self.best_iter = 0
        runner.should_stop = False

    def after_val_epoch(self, runner):
        if self.metric in runner.log_buffer.output:
            current_score = runner.log_buffer.output[self.metric]

            if self.rule == 'greater':
                improved = current_score > self.best_score + self.min_delta
            else:
                improved = current_score < self.best_score - self.min_delta

            if improved:
                self.best_score = current_score
                self.best_iter = runner.iter
                self.counter = 0
            else:
                self.counter += 1

            runner.logger.info(f"[EarlyStopping] 当前 {self.metric}: {current_score:.4f}, 最佳: {self.best_score:.4f}, Counter: {self.counter}/{self.patience}")

            if self.counter >= self.patience:
                runner.logger.info(
                    f'✅ 提前停止训练：{self.metric} 在 {self.patience} 次验证中无改进。\n'
                    f'📌 最佳 {self.metric}: {self.best_score:.4f}，出现在迭代 {self.best_iter}'
                )
                runner._max_iters = runner.iter
