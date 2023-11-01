if [ ! -e ../vae/models/MNIST_EnD.pth ];                                        
then                                                                            
    cd ../vae
    mkdir -p models 
    python train.py                                                             
fi

cd ../stunted_dataset
mkdir -p models
sh pollute_train.sh
