#!/usr/bin/env bash

mkdir -p mnist
wget -O mnist/train-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -O mnist/train-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -O mnist/t10k-images-idx3-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -O mnist/t10k-labels-idx1-ubyte.gz http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

mkdir -p svhn
wget -O svhn/train_32x32.mat http://ufldl.stanford.edu/housenumbers/train_32x32.mat
wget -O svhn/test_32x32.mat http://ufldl.stanford.edu/housenumbers/test_32x32.mat
wget -O svhn/extra_32x32.mat http://ufldl.stanford.edu/housenumbers/extra_32x32.mat