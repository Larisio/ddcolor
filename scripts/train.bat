set CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=3721 basicsr/train.py -opt options/train/train_ddcolor.yml --auto_resume --launcher pytorch

