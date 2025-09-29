checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_3rdparty-fcmae_in1k_20230104-07514db4.pth'
crop_size = (
    1024,
    1024,
)
custom_imports = dict(allow_failed_imports=False, imports='mmpretrain.models')
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
dataset_type = 'KCQRCrackDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=1000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
kcqr_2023_v1 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v10 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v10_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터'
kcqr_2023_v1_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지'
kcqr_2023_v2 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v2_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋'
kcqr_2023_v3 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v3_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34'
kcqr_2023_v4 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v4_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis'
kcqr_2023_v5 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v5_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2'
kcqr_2023_v6 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v6_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1'
kcqr_2023_v7 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v7_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)'
kcqr_2023_v8 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v8_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터'
kcqr_2023_v9 = dict(
    test=dict(
        data_prefix=dict(
            img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    train=dict(
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(crop_size=(
                1024,
                1024,
            ), type='RandomCrop'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'),
    val=dict(
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        data_root=
        '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split',
        label_map=dict({
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0
        }),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='KCQRCrackDataset'))
kcqr_2023_v9_root = '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split'
label_map = dict({2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0})
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        arch='atto',
        drop_path_rate=0.4,
        gap_before_final_norm=False,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_3rdparty-fcmae_in1k_20230104-07514db4.pth',
            prefix='backbone.',
            type='Pretrained'),
        layer_scale_init_value=1.0,
        out_indices=[
            0,
            1,
            2,
            3,
        ],
        type='mmpretrain.ConvNeXt',
        use_grn=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            1024,
            1024,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=128,
        dropout_ratio=-1,
        feature_strides=[
            4,
            8,
            16,
            32,
        ],
        in_channels=[
            128,
            128,
            128,
            128,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=dict(
            class_weight=[
                10.0,
                20.0,
            ],
            loss_weight=1.0,
            type='CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        type='FPNHead'),
    neck=dict(
        in_channels=[
            40,
            80,
            160,
            320,
        ],
        num_outs=4,
        out_channels=128,
        type='FPN'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=6),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=80000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/test', seg_map_path='gtFine/test'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=120000, type='IterBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        keep_ratio=True,
                        ratio_range=(
                            0.5,
                            2.0,
                        ),
                        scale=(
                            1024,
                            1024,
                        ),
                        type='RandomResize'),
                    dict(prob=0.5, type='RandomFlip'),
                    dict(crop_size=(
                        1024,
                        1024,
                    ), type='RandomCrop'),
                    dict(type='PhotoMetricDistortion'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(crop_size=(
        1024,
        1024,
    ), type='RandomCrop'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        datasets=[
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2022_현장촬영이미지',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/2023_현장촬영이미지_학습용데이터셋',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/AI_Hub_콘크리트균열_원천_34',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/BKim_Thesis',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/genenral_crack_v0.1.2',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/general_crack_v0.1.1',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/project0811_학습데이터',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/한국도로공사_split',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
            dict(
                data_prefix=dict(
                    img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
                data_root=
                '/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206.hardnegative/SuperResolution/Crack/v0.1.5_학습데이터',
                label_map=dict({
                    2: 0,
                    3: 0,
                    4: 0,
                    5: 0,
                    6: 0,
                    7: 0,
                    8: 0
                }),
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        1024,
                        1024,
                    ), type='Resize'),
                    dict(type='LoadAnnotations'),
                    dict(type='PackSegInputs'),
                ],
                type='KCQRCrackDataset'),
        ],
        type='ConcatDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/SrPS800_Nonoversampling'
