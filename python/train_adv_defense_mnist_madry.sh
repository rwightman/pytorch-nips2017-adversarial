# As per https://github.com/MadryLab/mnist_challenge at commit 9f9bf84, 2017-10-21
#
# Epochs
# They run 100,000 iterations of 50 batch_size.
# 100,000 x 50 = 5,000,000 trained examples
# 5,000,000 trained examples / 60,000 mnist training examples per epoch = 83.33 training epochs

python train_adv_defense_mnist.py --epochs 83 --batch-size 50 --opt adam --lr 0.0001 --model-name madry --df