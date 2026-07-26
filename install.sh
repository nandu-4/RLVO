# setting conda hook
eval "$(conda shell.bash hook)"

# create env
conda create -y -n Real-LOD python=3.11
conda activate Real-LOD

# install pytorch
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# install dependencies of openmmlab
pip install -U openmim
mim install "mmengine==0.7.1"
mim install "mmcv==2.0.0rc4"
mim install "mmdet==3.3.0"

# install other dependencies
pip install -r requirements.txt

# install real-model
pip install -v -e .