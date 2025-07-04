conda create -n ddcolor python=3.10.18
conda activate ddcolor

# remove dlib from requirements.txt 
conda install -c conda-forge dlib

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org
pip install -r requirements.txt

python3.10 setup.py develop

# Download the pretrained weights for ConvNeXt and InceptionV3 and place them in the pretrain folder.
ConvNeXt: https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
Inception_v3: https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth



pip install modelscope