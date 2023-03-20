python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
--scale 0.01 --hidden-size 1

python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
--scale 0.01 --hidden-size 16

python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
--scale 0.01 --hidden-size 32

python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
--scale 0.01 --hidden-size 64

python main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22k.yaml \
--pretrained pretrained_models/swin_tiny_patch4_window7_224_22k.pth \
--data-path data/dataset --batch-size 8 --local_rank 0 --dataset DTD \
--scale 0.01 --hidden-size 256