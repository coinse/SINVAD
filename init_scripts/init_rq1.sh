cd ../sa
mkdir -p models
[ -e ../sa/models/MNIST_conv_classifier.pth ] || python train.py conv
[ -e ../sa/models/MNIST_classifier.pth ] || python train.py noconv
cd ../vae
mkdir -p models
[ -e ../vae/models/MNIST_EnD.pth ] || python train.py
