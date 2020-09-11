cd ../sa
python svhn_train.py vgg
python svhn_train.py mobile
python svhn_train.py custom
cd ../vae
python train_conv.py
