#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 64 --adapter full_model

#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset Aircraft \
#--scale 0.01 --hidden-size 64 --adapter full_model \
#--resume ./output/swin_tiny_patch4_window7_224_22k_0.01_64_Aircraft/default/ckpt_epoch_60.pth

#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset flower102 \
#--scale 0.01 --hidden-size 64 --adapter full_model

#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset stanford_cars \
#--scale 0.01 --hidden-size 64 --adapter full_model

#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset stanford_cars \
#--scale 0.01 --hidden-size 64 --adapter full_model \
#--resume ./output/swin_tiny_patch4_window7_224_22k_0.01_64_stanford_cars/default/ckpt_epoch_242.pth



python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 4 --local_rank 0 --dataset CIFAR100 \
--scale 0.01 --hidden-size 128 --adapter full_model \
--resume ./output/swin_tiny_patch4_window7_224_22k_0.01_16_CIFAR100/default/ckpt_epoch_7.pth
