if [ ! -e ../vae/models/MNIST_EnD.pth ];                                        
then                                                                            
    cd ../vae                                                                   
    python train.py                                                             
fi

cd ../stunted_dataset
sh pollute_train.sh
