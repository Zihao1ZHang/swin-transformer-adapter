# fine tuning full model
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD

# fine tuning model with parallel adapter
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 1 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 16 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 32 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 64 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 256 --adapter parallel

## fine tuning model with sequential adapter
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 1 --adapter seq
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 16 --adapter seq
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 32 --adapter seq
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 64 --adapter seq

## DTD
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 64 --adapter seq
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
#--scale 0.01 --hidden-size 64 --adapter parallel
#
## Aircraft
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset Aircraft \
#--scale 0.01 --hidden-size 64 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset Aircraft \
#--scale 0.01 --hidden-size 64 --adapter seq
#
## flowers
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset flower102 \
#--scale 0.01 --hidden-size 64 --adapter full_model
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset flower102 \
#--scale 0.01 --hidden-size 64 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset flower102 \
#--scale 0.01 --hidden-size 64 --adapter seq
#
### omniglot
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 4 --local_rank 0 --dataset omniglot \
#--scale 0.01 --hidden-size 64 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 4 --local_rank 0 --dataset omniglot \
#--scale 0.01 --hidden-size 64 --adapter seq
#
#
## stanford cars
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset stanford_cars \
#--scale 0.01 --hidden-size 64 --adapter parallel
#
#python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
#--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
#--data-path data/dataset --batch-size 8 --local_rank 0 --dataset stanford_cars \
#--scale 0.01 --hidden-size 64 --adapter seq


python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 6 --local_rank 0 --dataset CIFAR100 \
--scale 0.01 --hidden-size 64 --adapter seq

python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 6 --local_rank 0 --dataset CIFAR100 \
--scale 0.01 --hidden-size 64 --adapter parallel



python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 6 --local_rank 0 --dataset CIFAR100 \
--scale 0.01 --hidden-size 64 --adapter parallel \
--resume ./output/swin_tiny_patch4_window7_224_22k_0.01_64_CIFAR100/default/ckpt_epoch_161.pth

