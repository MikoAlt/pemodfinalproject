Miko : fixed jumlah neuron, variasi depth layer,
Yur : variasi jumlah neuron, depth layer fixed
Farrij : variasiin optimizer
Naryama : learning rate

Default setting model : depth layer 3, jumlah neuron 3200, susunan layer input->mlp->mlp->mlp->softmax, activation function leaky relu, inisialisasi parameter He, train test split 80/20, susunan neuron (layer 1: 70%, layer 2: 20%, layer 3: 10%), optimizer adamw, learning rate 0.001, weight decay 0.01, momentum (beta1 0.9), variance (beta2 0.99), epsilon 1e-8, epoch 100.

dataset : 40x40 greyscale image, preprocess terlebih dahulu, training data 80% dan test data 20%.

jadi akan satu model, tetapi metode pembelajaran nya berbeda. satu menggunakan vector based calculation, satu menggunakan matrix based calculation. vector based calculation menggunakan pure python / cupy tapi menggunakan vector dan bukan matriks, matrix based calculation menggunakan numpy / cupy. 