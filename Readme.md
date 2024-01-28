# Readme

## 1. Installation

This code has been run on a MacBook Pro with M1 Pro and an Ubuntu 20.04 machine with CUDA 11.8 and a Nvidia RTX 4090 with 24GB of VRAM.

Simply create a Conda environment with:

```bash
conda create -f environment.yml
conda activate cv-nerf
```

## 2. Training

Use

```
python train.py
```

You can specify hyperparamers including but not restricted to the dataset path, the batch size, the positional encoding frequency, etc. The command above will list all possible arguments.

The checkpoints will be saved at `checkpoints/{name}`, and testing results during training will be saved at `runs/{name}/train`.

## 3. Testing

Use

```
python test.py
```

Much like in training, you can also specify arguments. You can find the testing results at `runs/{name}/train`.

## 4. Testing Depths
Use
```
python test_depth.py
```
You can also specify arguments.