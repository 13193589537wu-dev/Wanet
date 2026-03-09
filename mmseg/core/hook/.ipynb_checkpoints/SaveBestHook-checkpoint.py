from mmcv.runner import Hook, HOOKS
import os
import torch

@HOOKS.register_module()
class SaveBestCheckpointHook(Hook):
    def __init__(self, save_dir, metric='mIoU', rule='greater', max_keep_ckpts=1):
        print('🟢 SaveBestCheckpointHook: instance created')
        self.save_dir = save_dir
        self.metric = metric
        self.rule = rule
        self.best_metric = None
        self.max_keep_ckpts = max_keep_ckpts
        self.ckpt_list = []

        os.makedirs(self.save_dir, exist_ok=True)
        print(f'[Hook] Initialized SaveBestCheckpointHook, watching metric: {self.metric}')

    def after_val_epoch(self, runner):
        log_dict = runner.log_buffer.output

        if self.metric not in log_dict:
            print(f'[Hook] Warning: Metric "{self.metric}" not found in log_buffer.output')
            return

        current_metric = log_dict[self.metric]

        is_better = (
            self.best_metric is None or
            (self.rule == 'greater' and current_metric > self.best_metric) or
            (self.rule == 'less' and current_metric < self.best_metric)
        )

        if is_better:
            self.best_metric = current_metric
            self._save_checkpoint(runner)

    def _save_checkpoint(self, runner):
        iter_num = runner.iter
        filename = f'best_{self.metric}_iter_{iter_num}.pth'
        checkpoint_path = os.path.join(self.save_dir, filename)

        try:
            runner.save_checkpoint(
                out_dir=self.save_dir,
                filename_tmpl=filename,
                meta={'best_metric': self.best_metric}
            )
            print(f'[Hook] ✅ Saved best checkpoint at iter {iter_num}: {self.best_metric:.4f}')
        except Exception as e:
            print(f'[Hook] ❌ Failed to save checkpoint: {e}')
            return

        self.ckpt_list.append(filename)
        if len(self.ckpt_list) > self.max_keep_ckpts:
            old_ckpt = self.ckpt_list.pop(0)
            old_ckpt_path = os.path.join(self.save_dir, old_ckpt)
            try:
                os.remove(old_ckpt_path)
                print(f'[Hook] 🗑️ Removed old checkpoint: {old_ckpt}')
            except Exception as e:
                print(f'[Hook] ⚠️ Failed to remove old checkpoint: {e}')
