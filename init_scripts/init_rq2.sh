if [ ! -e ../vae/models/MNIST_EnD.pth ];
then
    cd ../vae
    python train.py
fi

if [ ! -e ../sa/models/MNIST_conv_classifier.pth ];
then
    cd ../sa
    python train.py conv
fi

cd ..
python gen_bound_imgs.py
