if [ ! -e ../vae/models/MNIST_EnD.pth ];
then
    cd ../vae
    mkdir -p models
    python train.py
fi

if [ ! -e ../sa/models/MNIST_conv_classifier.pth ];
then
    cd ../sa
    mkdir -p models
    python train.py conv
fi

cd ..
python gen_bound_imgs.py
python label_bound_imgs.py
