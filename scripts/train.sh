CUDA_VISIBLE_DEVICES=0
#python -m torch.distributed.launch --nproc_per_node=4 --master_port=3721 basicsr/train.py -opt options/train/train_ddcolor.yml --auto_resume --launcher pytorch
torchrun --nproc_per_node=1 --master_port=3721 /home/lkmm73/ddcolor/basicsr/train.py -opt  /home/lkmm73/ddcolor/options/train/train_ddcolor.yml --auto_resume --launcher pytorch