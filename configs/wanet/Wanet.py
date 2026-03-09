custom_imports = dict(
    imports=['mmseg.core.hook.SaveBestHook'],  # 根据你的真实路径修改
    allow_failed_imports=False
)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder_Distill',
    pretrained=None,
    backbone=dict(
        type='WANet',
        init_cfg=None,
        base_channels=64,
        spp_channels=128),
    decode_head=dict(
        type='SCTHead',
        in_channels=256,
        channels=256,
        dropout_ratio=0.0,
        in_index=0,
        num_classes=2, #已修改
        align_corners=False,
       loss_decode =[
            dict(     #添加
                type='CrossEntropyLoss',
                use_sigmoid=False,  # 正确！保持 False，因为是多分类结构
                class_weight=[0.5, 3.0],    #[1.0, 3.0],   #[0.5, 3.0]
                loss_weight=1.0
            ),
            dict(
            type='DiceLoss',  # 使用DiceLoss
             use_sigmoid=False,  # 二分类的情况下，通常使用sigmoid
             loss_weight=3.0    # 可调节损失权重   原3.0
            )
           ]),
    auxiliary_head=[
        dict(
            type='AU_SCTHead',
            in_channels=128,
            channels=128,
            dropout_ratio=0.0,
            in_index=1,
            num_classes=2,          #已修改
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
               # class_weight=[1.0, 1.0],
                class_weight=[0.5, 3.0],     #增大预测类的权重  原[0.5,1.5]
                loss_weight=1.0)),         #原损失值 0.4
        dict(
            type='VitGuidanceHead',
            init_cfg=None,
            in_channels=256,
            channels=256,
            base_channels=64,
            in_index=2,
            num_classes=1,                    
            loss_decode=dict(type='AlignmentLoss', loss_weight=[1.0, 1.0, 0.8, 0.6]))  
    ],                                                                       #[0.3, 0.6, 0.6, 0.6] 或 [0.2, 0.5, 0.5, 0.5]
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


#data
dataset_type = 'CustomBinaryDataset'      #修改 ADE20KDataset
data_root = '../ditch_data'    #换数据集只需改这里
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)    #图像分辨率大小
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),     #已修改
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.75, 1.25)),  #2048, 512   0.5, 2.0
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.25),    # cat_max_ratio=0.75   已修改图像分辨率
    dict(type='RandomFlip', prob=0),  #0.5
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),  # 512，512
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
   # dict(type='LoadAnnotations', reduce_zero_label=False),  # 已添加#修复get
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),    #2048, 512
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),

        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CustomBinaryDataset',    #已修改
        data_root='../ditch_data',
        img_dir='images/training',
        ann_dir='annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),   #已修改
            dict(type='Resize', img_scale=(512, 512), ratio_range=(0.75, 1.25)),   #原 2048, 512
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.25),     #cat_max_ratio=0.75
            dict(type='RandomFlip', prob=0),  #0.5
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='CustomBinaryDataset',   #已修改
        data_root='../ditch_data',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CustomBinaryDataset',     #已修改
        data_root='../ditch_data',
        img_dir='images/testing',
        ann_dir='annotations/testing',
        pipeline=[
            dict(type='LoadImageFromFile'),
           # dict(type='LoadAnnotations', reduce_zero_label=False),  # 已添加#修复get
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512), #2048, 512
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                   # dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                    #dict(type='Collect', keys=['img', 'gt_semantic_seg'])
                ])
        ]))

#3.optimizer
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False),
                       dict(type='TensorboardLoggerHook')  # 新增：支持 TensorBoard 可视化
                       ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from =None  
resume_from =None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=0.0001,  #0.0005  0.000125   降低学习率
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            teacher_backbone=dict(lr_mult=0.0),
            teacher_head=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=5000,     #  原1500  3000 16万
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=81000)    #160000 60000     实验发现60000已经是最好的了   80000
#checkpoint_config = dict(by_epoch=False, interval=4000)    #添加, save_best='mDice'

evaluation = dict(interval=3000,metric=['mIoU', 'mDice', 'mFscore'], pre_eval=True)   #源代码: metric='mIoU'   原性能评估 8000
find_unused_parameters = True
auto_resume = False
seed = 1440161127
work_dir = "./Ditch_work_dirs/model/xiaobo/high"

checkpoint_config = dict(
    by_epoch=False,
    interval=3000,         # 每4000次迭代保存一次
    max_keep_ckpts=6     #新加
)

# 替换为标准的评估钩子（EvalHook）或移除自定义钩子
custom_hooks = [
    dict(
        type='EmptyCacheHook',       # 可选：清理GPU缓存，防止内存泄漏
        after_epoch=True,
        after_iter=True
    ),
    dict(                             # 新增：早停 Hook
        type='EarlyStoppingHook',
        patience=5,  # 连续5次评估没有提升就停止
        metric='mIoU',  # 以 mDice 为监控指标
        rule='greater'  # 指标越大越好
    ),
    dict(  # 自定义：保存最佳模型
            type="SaveBestCheckpointHook",
            save_dir='./work_dirs/my_model',  # 模型保存路径
            metric='mIoU',  # 监控的评估指标
            rule='greater',  # 若评估指标更高则保存
            max_keep_ckpts=2  # 最多保留3个最好的检查点
        )
]
