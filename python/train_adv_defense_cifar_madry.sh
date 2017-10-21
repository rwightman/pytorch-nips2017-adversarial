# As per https://github.com/MadryLab/cifar10_challenge at commit 16a265d, 2017-10-21
#
# Epochs
# They run 80,000 iterations of 128 batch_size.
# 80,000 x 128 = 10,240,000 trained examples
# 10,240,000 trained examples / 60,000 cifar training examples per epoch = 170.6 training epochs

python train_adv_defense_cifar.py --epochs 171 --batch-size 128 --opt sgd --lr 0.1 --model-name wr28x10 --df