from mmseg.datasets.custom import CustomDataset
#from mmseg.registry import DATASETS
from mmseg.datasets.builder import DATASETS
# import mmcv  # 添加此行
import os.path as osp
@DATASETS.register_module()    
class CustomBinaryDataset(CustomDataset):
    CLASSES = ('background', 'laker')  # 只有两类
    PALETTE = [[0, 0, 0], [255, 255, 255]]  # 黑白掩码

    def __init__(self, **kwargs):
        super(CustomBinaryDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,  # 这非常关键！标签0代表背景，1代表前景
            classes=('background', 'laker'),
            palette=[[0, 0, 0], [255, 255, 255]],
            **kwargs)

        assert osp.exists(self.img_dir)
#

    # def prepare_test_img(self, idx):
    #     img_info = self.img_infos[idx]
    #     results = {
    #         'img_info': dict(img_info),  # 确保是原生 dict
    #         'img_prefix': str(self.img_dir),  # 强制转字符串
    #         'filename': str(img_info['filename']),
    #         'seg_fields': [],
    #     }
    #
    #     if self.ann_dir is not None and 'ann' in img_info:
    #         results['ann_info'] = {
    #             'seg_map': str(osp.join(self.ann_dir, img_info['ann']['seg_map']))
    #         }
    #         results['seg_fields'] = ['gt_semantic_seg']
    #
    #     # 执行 pipeline 后强制转换结果
    #     processed = self.pipeline(results)
    #     return {
    #         'img': processed['img'],  # 假设 pipeline 返回的 key 是 'img'
    #         'gt_semantic_seg': processed.get('gt_semantic_seg', None)  # 标注可能不存在
    #     }

    # def load_annotations(self, img_dir=None, img_suffix=None, ann_dir=None, seg_map_suffix=None, split=None):
#         """Load annotation file paths."""
#         img_infos = []
#         if split is not None:
#             split_path = osp.join(self.data_root, split)
#             assert osp.exists(split_path), f"Split file {split_path} does not exist"
#             with open(split_path) as f:
#                 lines = f.read().strip().splitlines()
#             for line in lines:
#                 img_name = line.strip()
#                 img_info = dict(
#                     filename=img_name + self.img_suffix,
#                     ann_info=dict(seg_map=img_name + self.seg_map_suffix)
#                 )
#                 img_infos.append(img_info)
#         else:
#             img_dir = osp.join(self.data_root, self.img_dir)
#             assert osp.exists(img_dir), f"Image directory {img_dir} does not exist"
#             for img_file in mmcv.scandir(img_dir, self.img_suffix):
#                 img_name = osp.splitext(img_file)[0]
#                 img_info = dict(
#                     filename=img_file,
#                     ann_info=dict(seg_map=img_name + self.seg_map_suffix)
#                 )
#                 img_infos.append(img_info)
#         print(f"Loaded {len(img_infos)} images")
#         return img_infos
