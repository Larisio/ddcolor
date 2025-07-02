CUDA_VISIBLE_DEVICES=0 \
python infer.py --input ./assets/test_images --output ./results --model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt
#python infer.py --input ./assets/test_images --output ./results --model_path experiments/train_ddcolor_l_archived_20250619_142347/models/net_g_latest.pth