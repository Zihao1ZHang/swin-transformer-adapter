## Create directory for datasets
```angular2html
cd data
mkdir dataset
```
## Download the DTD dataset and put it into the /data/dataset folder
https://pytorch.org/vision/stable/_modules/torchvision/datasets/dtd.html#DTD

notice: the version of torchvision for the model does not contain the api for the DTD dataset

## Create directory for pretrained models
```angular2html
mkdir pretrained_models
```

## Download the pretrained model
https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth

## Fine-tuning the model
```angular2html
python main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 8 --accumulation-steps 2 --local_rank 0 --dataset DTD
```