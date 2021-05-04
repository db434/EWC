# Elastic Weight Consolidation

A TensorFlow v2 + Keras implementation of [Elastic Weight Consolidation](https://arxiv.org/abs/1612.00796).

The goal is to train a neural network on separate tasks sequentially, while minimising the loss of performance on older tasks.

## The story
When a neural network is trained on a dataset, we cannot expect it to perform well on another dataset with different properties. However, if we attempt to naively extend the network's training with a new dataset, it will quickly forget about what it learned previously. This problem is called *catastrophic forgetting*.

```commandline
python3 main.py --epochs=15 --splits=3 --dataset-update=permute
```

EWC addresses this problem by estimating how important each network parameter is in producing the final output, and restricting changes to parameters which are considered important.

```commandline
python3 main.py --epochs=15 --splits=3 --dataset-update=permute --ewc --ewc-lambda=0.01
```

I provide a variant of EWC called *Fisher information masking* (unpublished?). Instead of scaling the amount each parameter can change, updates are either blocked entirely or allowed to proceed as normal. This has similar strengths and weaknesses to EWC, but uses much less memory, and scales better to more tasks.

```commandline
python3 main.py --epochs=15 --splits=3 --dataset-update=permute --fim --fim-threshold=1e-6
```

While EWC is good at accommodating new inputs for classes it already knows about, it is still prone to catastrophic forgetting in incremental learning situations where whole classes are added/removed from the dataset.

```commandline
python3 main.py --epochs=15 --splits=3 --dataset-update=switch --ewc --ewc-lambda=0.01
```

One reason for EWC's limitations is that it uses a quadratic loss term. This can grow very large and make training unstable. I provide an implementation of gradient clipping from [IncDet](https://ieeexplore.ieee.org/document/9127478), which clips gradients to a linear function after a given threshold. 

```commandline
python3 main.py --epochs=15 --splits=3 --dataset-update=switch --ewc --ewc-lambda=0.01 --incdet --incdet-threshold=1e-6
```

I haven't seen much benefit from using gradient clipping in the simple examples here. The original paper also included a distillation loss term, which may help, but that is getting very close to [iCaRL](https://arxiv.org/abs/1611.07725), which is beyond the scope of this project.

## Usage

| Parameter | Default | Description | 
| --- | --- | --- |
| `--batch-size` | 256 | Number of inputs to process simultaneously. |
| `--epochs` | 20 | Number of iterations through training dataset. |
| `--learning-rate` | 0.001 | Initial learning rate for Adam optimiser. |
| `--model` | "mlp" | Neural network to train. Options: "mlp", simple multi-layer perceptron for MNIST; "lenet", simple CNN for MNIST; "cifarnet", simple CNN for CIFAR-10 and CIFAR-100. |
| `--dataset` | (Determined by model selected) | Dataset to use. Choices: "mnist", "cifar10", "cifar100". |
| `--dataset-update` | "full" | How to change the dataset during training. Options: "full", use the whole dataset throughout; "permute", apply a permutation to pixels; "increment", start with few classes and add more and more; "switch", start with few classes and switch to different ones. |
| `--splits` | 5 | Number of dataset partitions/permutations to create. |
| `--ewc` | | Enable EWC to preserve accuracy on previous datasets. |
| `--ewc-lambda` | 0.1 | Relative importance of old tasks vs new tasks. Higher values favour old tasks. |
| `--ewc-samples` | 100 | Number of dataset samples used to estimate weight importance for EWC. More samples means better estimates. |
| `--fim` | | Enable Fisher information masking to preserve accuracy on previous datasets. |
| `--fim-threshold` | 1e-6 | Threshold controlling when to freeze weights. Lower thresholds favour old tasks. |
| `--fim-samples` | 100 | Number of dataset samples used to estimate weight importance for FIM. More samples means better estimates. |
| `--incdet` | | Enable IncDet's gradient clipping to stabilise EWC training. |
| `--incdet-threshold` | 1e-6 | Threshold for IncDet gradient clipping. Lower values favour old tasks. |