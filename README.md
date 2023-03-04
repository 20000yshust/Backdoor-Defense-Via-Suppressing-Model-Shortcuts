# Backdoor-Defense-Via-Suppressing-Model-Shortcuts

This is the official implementation of our paper Backdoor Defense Via Suppressing Model Shortcuts.

python train_backdoormodel.py --model=resnet18 --datasets=cifar10 --attack=badnets --target=1 --poisoned_rate=0.05 --mp=experiments --gpu=0

python chooselayer.py --model==resnet18 --datasets=cifar10 --model_path=filepath --pdp=filepath --gpu=0
