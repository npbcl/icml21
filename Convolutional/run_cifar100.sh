### !/bin/sh
rm -r cache
rm -r saves
### Discriminative Experiments
# not MNIST
echo "Training Cifar100"
python3 npbcl_cifar100.py
echo "saves\ncache/not_mnist" | python3 save.py
cp all_masks.png cache/not_mnist
cp union_mask.png cache/not_mnist
rm -r saves