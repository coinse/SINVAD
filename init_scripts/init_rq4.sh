cd ../sa
mkdir -p models
python svhn_train.py vgg
python svhn_train.py mobile
python svhn_train.py custom
cd ../vae
mkdir -p models
python train_conv.py
