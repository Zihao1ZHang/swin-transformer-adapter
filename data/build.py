# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation, str_to_pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler

from .DTD import DTD
from .FGVC_AIRCRAFT import FGVCAircraft
from .flowers102 import Flowers102
from .omniglot import Omniglot
from .stanford_cars import StanfordCars
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    elif config.DATA.DATASET == 'DTD':
        dataset = DTD(config.DATA.DATA_PATH, transform=transform)
        nb_classes = 47
    elif config.DATA.DATASET == 'Aircraft':
        dataset = FGVCAircraft(config.DATA.DATA_PATH, transform=transform)
        nb_classes = 100
    elif config.DATA.DATASET == 'flower102':
        dataset = Flowers102(config.DATA.DATA_PATH, transform=transform)
        nb_classes = 102
    elif config.DATA.DATASET == 'omniglot':
        dataset = Omniglot(config.DATA.DATA_PATH, transform=transform)
        nb_classes = 1623
    elif config.DATA.DATASET == 'stanford_cars':
        dataset = StanfordCars(config.DATA.DATA_PATH, transform=transform)
        nb_classes = 196
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if config.DATA.DATASET == 'omniglot' and is_train:

        t = []
        if resize_im:
            if config.TEST.CROP:
                size = int((256 / 224) * config.DATA.IMG_SIZE)
                t.append(
                    transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                    # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
            else:
                t.append(
                    transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                      interpolation=_pil_interp(config.DATA.INTERPOLATION))
                )

        # t.append(transforms.Grayscale(num_output_channels=3))
        t.append(transforms.ToTensor())
        if config.DATA.DATASET == 'omniglot':
            stack_fn = lambda x: torch.stack([x, x, x], dim=0)
            stack_transform = transforms.Lambda(stack_fn)
            squeeze_fn = lambda x: x.squeeze()
            squeeze_transform = transforms.Lambda(squeeze_fn)
            # cat_fn = lambda x: torch.cat([x, x, x], dim=0)
            # cat_transform = transforms.Lambda(cat_fn)
            t.append(squeeze_transform)
            t.append(stack_transform)


            # t.append(cat_transform)
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
        # scale = tuple((0.08, 1.0))  # default imagenet scale range
        # ratio = tuple((3. / 4., 4. / 3.))  # default imagenet ratio range
        # primary_tfl = [
        #     RandomResizedCropAndInterpolation(config.DATA.IMG_SIZE, scale=scale, ratio=ratio, interpolation=config.DATA.INTERPOLATION)]
        # primary_tfl += [transforms.RandomHorizontalFlip(p=0.5)]
        #
        # secondary_tfl = []
        # disable_color_jitter = False
        # if config.AUG.AUTO_AUGMENT:
        #     assert isinstance(config.AUG.AUTO_AUGMENT, str)
        #     # color jitter is typically disabled if AA/RA on,
        #     # this allows override without breaking old hparm cfgs
        #     disable_color_jitter = not (False or '3a' in config.AUG.AUTO_AUGMENT)
        #     if isinstance(config.DATA.IMG_SIZE, (tuple, list)):
        #         img_size_min = min(config.DATA.IMG_SIZE)
        #     else:
        #         img_size_min = config.DATA.IMG_SIZE
        #     aa_params = dict(
        #         translate_const=int(img_size_min * 0.45),
        #         img_mean=tuple([min(255, round(255 * x)) for x in IMAGENET_DEFAULT_MEAN]),
        #     )
        #     if config.DATA.INTERPOLATION and config.DATA.INTERPOLATION != 'random':
        #         aa_params['interpolation'] = str_to_pil_interp(config.DATA.INTERPOLATION)
        #     if config.AUG.AUTO_AUGMENT.startswith('rand'):
        #         secondary_tfl += [rand_augment_transform(config.AUG.AUTO_AUGMENT, aa_params)]
        #
        # if config.AUG.COLOR_JITTER is not None and not disable_color_jitter:
        #     color_jitter = (float(config.AUG.COLOR_JITTER),) * 3
        #     secondary_tfl += [transforms.ColorJitter(*color_jitter)]
        #
        # final_tfl = []
        # stack_fn = lambda x: torch.stack([x, x, x], dim=0)
        # stack_transform = transforms.Lambda(stack_fn)
        # squeeze_fn = lambda x: x.squeeze()
        # squeeze_transform = transforms.Lambda(squeeze_fn)
        # final_tfl += [
        #     transforms.ToTensor(),
        #     squeeze_transform,
        #     stack_transform,
        #     transforms.Normalize(
        #         mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
        #         std=torch.tensor(IMAGENET_DEFAULT_STD))
        # ]
        # if config.AUG.REPROB > 0.:
        #     final_tfl.append(
        #         RandomErasing(config.AUG.REPROB, mode=config.AUG.REMODE, max_count=config.AUG.RECOUNT,
        #                       num_splits=0, device='cpu'))
        # return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None, # 0.4
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None, #'rand-m9-mstd0.5-inc1'
            re_prob=config.AUG.REPROB, # 0.25
            re_mode=config.AUG.REMODE, # pixel
            re_count=config.AUG.RECOUNT, # 1
            interpolation=config.DATA.INTERPOLATION, # bicubic
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        if config.DATA.DATASET == 'omniglot':
            stack_fn = lambda x: torch.stack([x, x, x], dim=0)
            stack_transform = transforms.Lambda(stack_fn)
            squeeze_fn = lambda x: x.squeeze()
            squeeze_transform = transforms.Lambda(squeeze_fn)
            # cat_fn = lambda x: torch.cat([x, x, x], dim=0)
            # cat_transform = transforms.Lambda(cat_fn)
            transform2 = transforms.Compose([squeeze_transform, stack_transform])
            return transforms.Compose(transform.transforms + transform2.transforms)
            # return (transform + transform2)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    # t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.ToTensor())
    if config.DATA.DATASET == 'omniglot':
        stack_fn = lambda x: torch.stack([x, x, x], dim=0)
        stack_transform = transforms.Lambda(stack_fn)
        squeeze_fn = lambda x: x.squeeze()
        squeeze_transform = transforms.Lambda(squeeze_fn)
        # cat_fn = lambda x: torch.cat([x, x, x], dim=0)
        # cat_transform = transforms.Lambda(cat_fn)
        t.append(squeeze_transform)
        t.append(stack_transform)
        # t.append(cat_transform)
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
