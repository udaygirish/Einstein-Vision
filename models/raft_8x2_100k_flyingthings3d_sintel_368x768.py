checkpoint_config = dict(by_epoch=False, interval=10000)
crop_size = (
    368,
    768,
)
data = dict(
    test=dict(
        datasets=[
            dict(
                data_root='data/Sintel',
                pass_style='clean',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(exponent=3, type='InputPad'),
                    dict(
                        mean=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        std=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(type='TestFormatBundle'),
                    dict(
                        keys=[
                            'imgs',
                        ],
                        meta_keys=[
                            'flow_gt',
                            'filename1',
                            'filename2',
                            'ori_filename1',
                            'ori_filename2',
                            'ori_shape',
                            'img_shape',
                            'img_norm_cfg',
                            'scale_factor',
                            'pad_shape',
                            'pad',
                        ],
                        type='Collect'),
                ],
                test_mode=True,
                type='Sintel'),
            dict(
                data_root='data/Sintel',
                pass_style='final',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(exponent=3, type='InputPad'),
                    dict(
                        mean=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        std=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(type='TestFormatBundle'),
                    dict(
                        keys=[
                            'imgs',
                        ],
                        meta_keys=[
                            'flow_gt',
                            'filename1',
                            'filename2',
                            'ori_filename1',
                            'ori_filename2',
                            'ori_shape',
                            'img_shape',
                            'img_norm_cfg',
                            'scale_factor',
                            'pad_shape',
                            'pad',
                        ],
                        type='Collect'),
                ],
                test_mode=True,
                type='Sintel'),
        ],
        separate_eval=True,
        type='ConcatDataset'),
    test_dataloader=dict(samples_per_gpu=1, shuffle=False, workers_per_gpu=2),
    train=[
        dict(
            dataset=dict(
                data_root='data/Sintel',
                pass_style='clean',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        asymmetric_prob=0.2,
                        brightness=0.4,
                        contrast=0.4,
                        hue=0.1592356687898089,
                        saturation=0.4,
                        type='ColorJitter'),
                    dict(
                        bounds=[
                            50,
                            100,
                        ], max_num=3, prob=0.5, type='Erase'),
                    dict(
                        crop_size=(
                            368,
                            768,
                        ),
                        max_scale=0.6,
                        max_stretch=0.2,
                        min_scale=-0.2,
                        spacial_prob=0.8,
                        stretch_prob=0.8,
                        type='SpacialTransform'),
                    dict(crop_size=(
                        368,
                        768,
                    ), type='RandomCrop'),
                    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
                    dict(direction='vertical', prob=0.1, type='RandomFlip'),
                    dict(max_flow=1000.0, type='Validation'),
                    dict(
                        mean=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        std=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        keys=[
                            'imgs',
                            'flow_gt',
                            'valid',
                        ],
                        meta_keys=[
                            'filename1',
                            'filename2',
                            'ori_filename1',
                            'ori_filename2',
                            'filename_flow',
                            'ori_filename_flow',
                            'ori_shape',
                            'img_shape',
                            'erase_bounds',
                            'erase_num',
                            'scale_factor',
                        ],
                        type='Collect'),
                ],
                test_mode=False,
                type='Sintel'),
            times=100,
            type='RepeatDataset'),
        dict(
            dataset=dict(
                data_root='data/Sintel',
                pass_style='final',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        asymmetric_prob=0.2,
                        brightness=0.4,
                        contrast=0.4,
                        hue=0.1592356687898089,
                        saturation=0.4,
                        type='ColorJitter'),
                    dict(
                        bounds=[
                            50,
                            100,
                        ], max_num=3, prob=0.5, type='Erase'),
                    dict(
                        crop_size=(
                            368,
                            768,
                        ),
                        max_scale=0.6,
                        max_stretch=0.2,
                        min_scale=-0.2,
                        spacial_prob=0.8,
                        stretch_prob=0.8,
                        type='SpacialTransform'),
                    dict(crop_size=(
                        368,
                        768,
                    ), type='RandomCrop'),
                    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
                    dict(direction='vertical', prob=0.1, type='RandomFlip'),
                    dict(max_flow=1000.0, type='Validation'),
                    dict(
                        mean=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        std=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        keys=[
                            'imgs',
                            'flow_gt',
                            'valid',
                        ],
                        meta_keys=[
                            'filename1',
                            'filename2',
                            'ori_filename1',
                            'ori_filename2',
                            'filename_flow',
                            'ori_filename_flow',
                            'ori_shape',
                            'img_shape',
                            'erase_bounds',
                            'erase_num',
                            'scale_factor',
                        ],
                        type='Collect'),
                ],
                test_mode=False,
                type='Sintel'),
            times=100,
            type='RepeatDataset'),
        dict(
            data_root='data/flyingthings3d',
            pass_style='clean',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    asymmetric_prob=0.2,
                    brightness=0.4,
                    contrast=0.4,
                    hue=0.1592356687898089,
                    saturation=0.4,
                    type='ColorJitter'),
                dict(bounds=[
                    50,
                    100,
                ], max_num=3, prob=0.5, type='Erase'),
                dict(
                    crop_size=(
                        368,
                        768,
                    ),
                    max_scale=0.8,
                    max_stretch=0.2,
                    min_scale=-0.4,
                    spacial_prob=0.8,
                    stretch_prob=0.8,
                    type='SpacialTransform'),
                dict(crop_size=(
                    368,
                    768,
                ), type='RandomCrop'),
                dict(direction='horizontal', prob=0.5, type='RandomFlip'),
                dict(direction='vertical', prob=0.1, type='RandomFlip'),
                dict(max_flow=1000.0, type='Validation'),
                dict(
                    mean=[
                        127.5,
                        127.5,
                        127.5,
                    ],
                    std=[
                        127.5,
                        127.5,
                        127.5,
                    ],
                    to_rgb=False,
                    type='Normalize'),
                dict(type='DefaultFormatBundle'),
                dict(
                    keys=[
                        'imgs',
                        'flow_gt',
                        'valid',
                    ],
                    meta_keys=[
                        'filename1',
                        'filename2',
                        'ori_filename1',
                        'ori_filename2',
                        'filename_flow',
                        'ori_filename_flow',
                        'ori_shape',
                        'img_shape',
                        'erase_bounds',
                        'erase_num',
                        'scale_factor',
                    ],
                    type='Collect'),
            ],
            scene='left',
            test_mode=False,
            type='FlyingThings3D'),
    ],
    train_dataloader=dict(
        drop_last=True,
        persistent_workers=True,
        samples_per_gpu=2,
        shuffle=True,
        workers_per_gpu=2),
    val=dict(
        datasets=[
            dict(
                data_root='data/Sintel',
                pass_style='clean',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(exponent=3, type='InputPad'),
                    dict(
                        mean=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        std=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(type='TestFormatBundle'),
                    dict(
                        keys=[
                            'imgs',
                        ],
                        meta_keys=[
                            'flow_gt',
                            'filename1',
                            'filename2',
                            'ori_filename1',
                            'ori_filename2',
                            'ori_shape',
                            'img_shape',
                            'img_norm_cfg',
                            'scale_factor',
                            'pad_shape',
                            'pad',
                        ],
                        type='Collect'),
                ],
                test_mode=True,
                type='Sintel'),
            dict(
                data_root='data/Sintel',
                pass_style='final',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(exponent=3, type='InputPad'),
                    dict(
                        mean=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        std=[
                            127.5,
                            127.5,
                            127.5,
                        ],
                        to_rgb=False,
                        type='Normalize'),
                    dict(type='TestFormatBundle'),
                    dict(
                        keys=[
                            'imgs',
                        ],
                        meta_keys=[
                            'flow_gt',
                            'filename1',
                            'filename2',
                            'ori_filename1',
                            'ori_filename2',
                            'ori_shape',
                            'img_shape',
                            'img_norm_cfg',
                            'scale_factor',
                            'pad_shape',
                            'pad',
                        ],
                        type='Collect'),
                ],
                test_mode=True,
                type='Sintel'),
        ],
        separate_eval=True,
        type='ConcatDataset'),
    val_dataloader=dict(
        persistent_workers=True,
        samples_per_gpu=1,
        shuffle=False,
        workers_per_gpu=2))
dist_params = dict(backend='nccl')
evaluation = dict(interval=10000, metric='EPE')
flyingthing3d_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        asymmetric_prob=0.2,
        brightness=0.4,
        contrast=0.4,
        hue=0.1592356687898089,
        saturation=0.4,
        type='ColorJitter'),
    dict(bounds=[
        50,
        100,
    ], max_num=3, prob=0.5, type='Erase'),
    dict(
        crop_size=(
            368,
            768,
        ),
        max_scale=0.8,
        max_stretch=0.2,
        min_scale=-0.4,
        spacial_prob=0.8,
        stretch_prob=0.8,
        type='SpacialTransform'),
    dict(crop_size=(
        368,
        768,
    ), type='RandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.1, type='RandomFlip'),
    dict(max_flow=1000.0, type='Validation'),
    dict(
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        to_rgb=False,
        type='Normalize'),
    dict(type='DefaultFormatBundle'),
    dict(
        keys=[
            'imgs',
            'flow_gt',
            'valid',
        ],
        meta_keys=[
            'filename1',
            'filename2',
            'ori_filename1',
            'ori_filename2',
            'filename_flow',
            'ori_filename_flow',
            'ori_shape',
            'img_shape',
            'erase_bounds',
            'erase_num',
            'scale_factor',
        ],
        type='Collect'),
]
flyingthings3d_clean_train = dict(
    data_root='data/flyingthings3d',
    pass_style='clean',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            hue=0.1592356687898089,
            saturation=0.4,
            type='ColorJitter'),
        dict(bounds=[
            50,
            100,
        ], max_num=3, prob=0.5, type='Erase'),
        dict(
            crop_size=(
                368,
                768,
            ),
            max_scale=0.8,
            max_stretch=0.2,
            min_scale=-0.4,
            spacial_prob=0.8,
            stretch_prob=0.8,
            type='SpacialTransform'),
        dict(crop_size=(
            368,
            768,
        ), type='RandomCrop'),
        dict(direction='horizontal', prob=0.5, type='RandomFlip'),
        dict(direction='vertical', prob=0.1, type='RandomFlip'),
        dict(max_flow=1000.0, type='Validation'),
        dict(
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            std=[
                127.5,
                127.5,
                127.5,
            ],
            to_rgb=False,
            type='Normalize'),
        dict(type='DefaultFormatBundle'),
        dict(
            keys=[
                'imgs',
                'flow_gt',
                'valid',
            ],
            meta_keys=[
                'filename1',
                'filename2',
                'ori_filename1',
                'ori_filename2',
                'filename_flow',
                'ori_filename_flow',
                'ori_shape',
                'img_shape',
                'erase_bounds',
                'erase_num',
                'scale_factor',
            ],
            type='Collect'),
    ],
    scene='left',
    test_mode=False,
    type='FlyingThings3D')
img_norm_cfg = dict(
    mean=[
        127.5,
        127.5,
        127.5,
    ], std=[
        127.5,
        127.5,
        127.5,
    ], to_rgb=False)
load_from = 'https://download.openmmlab.com/mmflow/raft/raft_8x2_100k_flyingthings3d_400x720.pth'
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=50)
log_level = 'INFO'
lr_config = dict(
    anneal_strategy='linear',
    max_lr=0.000125,
    pct_start=0.05,
    policy='OneCycle',
    total_steps=100100)
model = dict(
    cxt_channels=128,
    cxt_encoder=dict(
        in_channels=3,
        init_cfg=[
            dict(
                layer=[
                    'Conv2d',
                ],
                mode='fan_out',
                nonlinearity='relu',
                type='Kaiming'),
            dict(bias=0, layer=[
                'SyncBatchNorm2d',
            ], type='Constant', val=1),
        ],
        net_type='Basic',
        norm_cfg=dict(type='SyncBN'),
        out_channels=256,
        type='RAFTEncoder'),
    decoder=dict(
        act_cfg=dict(type='ReLU'),
        corr_op_cfg=dict(align_corners=True, type='CorrLookup'),
        flow_loss=dict(gamma=0.85, type='SequenceLoss'),
        gru_type='SeqConv',
        iters=12,
        net_type='Basic',
        num_levels=4,
        radius=4,
        type='RAFTDecoder'),
    encoder=dict(
        in_channels=3,
        init_cfg=[
            dict(
                layer=[
                    'Conv2d',
                ],
                mode='fan_out',
                nonlinearity='relu',
                type='Kaiming'),
            dict(bias=0, layer=[
                'InstanceNorm2d',
            ], type='Constant', val=1),
        ],
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        out_channels=256,
        type='RAFTEncoder'),
    freeze_bn=True,
    h_channels=128,
    num_levels=4,
    radius=4,
    test_cfg=dict(iters=32),
    train_cfg=dict(),
    type='RAFT')
optimizer = dict(
    amsgrad=False,
    betas=(
        0.9,
        0.999,
    ),
    eps=1e-08,
    lr=0.000125,
    type='AdamW',
    weight_decay=1e-05)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
resume_from = None
runner = dict(max_iters=100000, type='IterBasedRunner')
sintel_clean_test = dict(
    data_root='data/Sintel',
    pass_style='clean',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(exponent=3, type='InputPad'),
        dict(
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            std=[
                127.5,
                127.5,
                127.5,
            ],
            to_rgb=False,
            type='Normalize'),
        dict(type='TestFormatBundle'),
        dict(
            keys=[
                'imgs',
            ],
            meta_keys=[
                'flow_gt',
                'filename1',
                'filename2',
                'ori_filename1',
                'ori_filename2',
                'ori_shape',
                'img_shape',
                'img_norm_cfg',
                'scale_factor',
                'pad_shape',
                'pad',
            ],
            type='Collect'),
    ],
    test_mode=True,
    type='Sintel')
sintel_clean_train = dict(
    data_root='data/Sintel',
    pass_style='clean',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            hue=0.1592356687898089,
            saturation=0.4,
            type='ColorJitter'),
        dict(bounds=[
            50,
            100,
        ], max_num=3, prob=0.5, type='Erase'),
        dict(
            crop_size=(
                368,
                768,
            ),
            max_scale=0.6,
            max_stretch=0.2,
            min_scale=-0.2,
            spacial_prob=0.8,
            stretch_prob=0.8,
            type='SpacialTransform'),
        dict(crop_size=(
            368,
            768,
        ), type='RandomCrop'),
        dict(direction='horizontal', prob=0.5, type='RandomFlip'),
        dict(direction='vertical', prob=0.1, type='RandomFlip'),
        dict(max_flow=1000.0, type='Validation'),
        dict(
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            std=[
                127.5,
                127.5,
                127.5,
            ],
            to_rgb=False,
            type='Normalize'),
        dict(type='DefaultFormatBundle'),
        dict(
            keys=[
                'imgs',
                'flow_gt',
                'valid',
            ],
            meta_keys=[
                'filename1',
                'filename2',
                'ori_filename1',
                'ori_filename2',
                'filename_flow',
                'ori_filename_flow',
                'ori_shape',
                'img_shape',
                'erase_bounds',
                'erase_num',
                'scale_factor',
            ],
            type='Collect'),
    ],
    test_mode=False,
    type='Sintel')
sintel_clean_train_x100 = dict(
    dataset=dict(
        data_root='data/Sintel',
        pass_style='clean',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                asymmetric_prob=0.2,
                brightness=0.4,
                contrast=0.4,
                hue=0.1592356687898089,
                saturation=0.4,
                type='ColorJitter'),
            dict(bounds=[
                50,
                100,
            ], max_num=3, prob=0.5, type='Erase'),
            dict(
                crop_size=(
                    368,
                    768,
                ),
                max_scale=0.6,
                max_stretch=0.2,
                min_scale=-0.2,
                spacial_prob=0.8,
                stretch_prob=0.8,
                type='SpacialTransform'),
            dict(crop_size=(
                368,
                768,
            ), type='RandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.1, type='RandomFlip'),
            dict(max_flow=1000.0, type='Validation'),
            dict(
                mean=[
                    127.5,
                    127.5,
                    127.5,
                ],
                std=[
                    127.5,
                    127.5,
                    127.5,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(type='DefaultFormatBundle'),
            dict(
                keys=[
                    'imgs',
                    'flow_gt',
                    'valid',
                ],
                meta_keys=[
                    'filename1',
                    'filename2',
                    'ori_filename1',
                    'ori_filename2',
                    'filename_flow',
                    'ori_filename_flow',
                    'ori_shape',
                    'img_shape',
                    'erase_bounds',
                    'erase_num',
                    'scale_factor',
                ],
                type='Collect'),
        ],
        test_mode=False,
        type='Sintel'),
    times=100,
    type='RepeatDataset')
sintel_final_test = dict(
    data_root='data/Sintel',
    pass_style='final',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(exponent=3, type='InputPad'),
        dict(
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            std=[
                127.5,
                127.5,
                127.5,
            ],
            to_rgb=False,
            type='Normalize'),
        dict(type='TestFormatBundle'),
        dict(
            keys=[
                'imgs',
            ],
            meta_keys=[
                'flow_gt',
                'filename1',
                'filename2',
                'ori_filename1',
                'ori_filename2',
                'ori_shape',
                'img_shape',
                'img_norm_cfg',
                'scale_factor',
                'pad_shape',
                'pad',
            ],
            type='Collect'),
    ],
    test_mode=True,
    type='Sintel')
sintel_final_train = dict(
    data_root='data/Sintel',
    pass_style='final',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            hue=0.1592356687898089,
            saturation=0.4,
            type='ColorJitter'),
        dict(bounds=[
            50,
            100,
        ], max_num=3, prob=0.5, type='Erase'),
        dict(
            crop_size=(
                368,
                768,
            ),
            max_scale=0.6,
            max_stretch=0.2,
            min_scale=-0.2,
            spacial_prob=0.8,
            stretch_prob=0.8,
            type='SpacialTransform'),
        dict(crop_size=(
            368,
            768,
        ), type='RandomCrop'),
        dict(direction='horizontal', prob=0.5, type='RandomFlip'),
        dict(direction='vertical', prob=0.1, type='RandomFlip'),
        dict(max_flow=1000.0, type='Validation'),
        dict(
            mean=[
                127.5,
                127.5,
                127.5,
            ],
            std=[
                127.5,
                127.5,
                127.5,
            ],
            to_rgb=False,
            type='Normalize'),
        dict(type='DefaultFormatBundle'),
        dict(
            keys=[
                'imgs',
                'flow_gt',
                'valid',
            ],
            meta_keys=[
                'filename1',
                'filename2',
                'ori_filename1',
                'ori_filename2',
                'filename_flow',
                'ori_filename_flow',
                'ori_shape',
                'img_shape',
                'erase_bounds',
                'erase_num',
                'scale_factor',
            ],
            type='Collect'),
    ],
    test_mode=False,
    type='Sintel')
sintel_final_train_x100 = dict(
    dataset=dict(
        data_root='data/Sintel',
        pass_style='final',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                asymmetric_prob=0.2,
                brightness=0.4,
                contrast=0.4,
                hue=0.1592356687898089,
                saturation=0.4,
                type='ColorJitter'),
            dict(bounds=[
                50,
                100,
            ], max_num=3, prob=0.5, type='Erase'),
            dict(
                crop_size=(
                    368,
                    768,
                ),
                max_scale=0.6,
                max_stretch=0.2,
                min_scale=-0.2,
                spacial_prob=0.8,
                stretch_prob=0.8,
                type='SpacialTransform'),
            dict(crop_size=(
                368,
                768,
            ), type='RandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.1, type='RandomFlip'),
            dict(max_flow=1000.0, type='Validation'),
            dict(
                mean=[
                    127.5,
                    127.5,
                    127.5,
                ],
                std=[
                    127.5,
                    127.5,
                    127.5,
                ],
                to_rgb=False,
                type='Normalize'),
            dict(type='DefaultFormatBundle'),
            dict(
                keys=[
                    'imgs',
                    'flow_gt',
                    'valid',
                ],
                meta_keys=[
                    'filename1',
                    'filename2',
                    'ori_filename1',
                    'ori_filename2',
                    'filename_flow',
                    'ori_filename_flow',
                    'ori_shape',
                    'img_shape',
                    'erase_bounds',
                    'erase_num',
                    'scale_factor',
                ],
                type='Collect'),
        ],
        test_mode=False,
        type='Sintel'),
    times=100,
    type='RepeatDataset')
sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(exponent=3, type='InputPad'),
    dict(
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        to_rgb=False,
        type='Normalize'),
    dict(type='TestFormatBundle'),
    dict(
        keys=[
            'imgs',
        ],
        meta_keys=[
            'flow_gt',
            'filename1',
            'filename2',
            'ori_filename1',
            'ori_filename2',
            'ori_shape',
            'img_shape',
            'img_norm_cfg',
            'scale_factor',
            'pad_shape',
            'pad',
        ],
        type='Collect'),
]
sintel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        asymmetric_prob=0.2,
        brightness=0.4,
        contrast=0.4,
        hue=0.1592356687898089,
        saturation=0.4,
        type='ColorJitter'),
    dict(bounds=[
        50,
        100,
    ], max_num=3, prob=0.5, type='Erase'),
    dict(
        crop_size=(
            368,
            768,
        ),
        max_scale=0.6,
        max_stretch=0.2,
        min_scale=-0.2,
        spacial_prob=0.8,
        stretch_prob=0.8,
        type='SpacialTransform'),
    dict(crop_size=(
        368,
        768,
    ), type='RandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.1, type='RandomFlip'),
    dict(max_flow=1000.0, type='Validation'),
    dict(
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        std=[
            127.5,
            127.5,
            127.5,
        ],
        to_rgb=False,
        type='Normalize'),
    dict(type='DefaultFormatBundle'),
    dict(
        keys=[
            'imgs',
            'flow_gt',
            'valid',
        ],
        meta_keys=[
            'filename1',
            'filename2',
            'ori_filename1',
            'ori_filename2',
            'filename_flow',
            'ori_filename_flow',
            'ori_shape',
            'img_shape',
            'erase_bounds',
            'erase_num',
            'scale_factor',
        ],
        type='Collect'),
]
workflow = [
    (
        'train',
        1,
    ),
]
