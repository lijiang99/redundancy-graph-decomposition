# Efficient Filter Pruning: Reducing Model Complexity through Redundancy Graph Decomposition

## Requirements

| python | pytorch      |
|--------|--------------|
| 3.8.10 | 1.11.0+cu113 |

## Running Code

### pre-training model

You can use `train.py` to train models from scratch on the CIFAR-10, CIFAR-100, and CUB-200 datasets for subsequent pruning experiments. Alternatively, you can directly employ the pretrained models provided [here](https://drive.google.com/drive/folders/1Y9x6OpJmK1nF0yzzIWHoa9JpnlxTXZkR?usp=sharing).

```
# example: python train.py --arch resnet56 --dataset cifar10 --epochs 200 --step-size 50
```

### generating pruning information

You can use `generate_prune_info.py` to generate pruning information for any model based on specified thresholds. For the convenience of reproducible experiments, we provide the extracted pruning information for all models [here](https://drive.google.com/drive/folders/1MNzYwDrXdNzKiFgw2V2EXn_0OJ4GaMSx?usp=sharing).

```
# example: python generate_prune_info.py --arch resnet56 --dataset cifar10 --threshold 0.7
```

### pruning and fine-tuning model

You can prune and fine-tune a model on the CIFAR-10 and CIFAR-100 dataset by

```
# example: python prune_cifar.py --arch resnet56 --dataset cifar10 --threshold 0.7
```

You can prune and fine-tune a model on the ImageNet (LSVRC-2012) dataset by

```
# example: python prune_imagenet.py --arch resnet50 --threshold 0.7
```

You can prune and fine-tune a model on the CUB-200 dataset by

```
# example: python prune_cub200.py --arch vgg16_bn --threshold 0.85
```

### completing all experiments

You can complete all the aforementioned experiments at once by

```bash
./run.sh cifa10   # for all models on CIFAR-10
./run.sh cifa100  # for all models on CIFAR-100
./run.sh cub200   # for all models on CUB-200
./run.sh imagenet # for all models on ImageNet
```

## Experimental Results

We provide pruned models, along with comprehensive fine-tuning logs, to verify and reproduce the experimental results.

### [CIFAR-10](https://drive.google.com/drive/folders/1l-uM7ly4BJ_bF3w7QcVTSiqkiFu73Iu9?usp=sharing)

<table>
<tr><td>Model</td><td>Threshold</td><td>Top-1 Acc. Loss (%)</td><td>FR (%)</td><td>PR (%)</td></tr>
<tr><td rowspan="3">VGGNet-16-BN</td><td>0.8</td><td>-0.14</td><td>42.14</td><td>76.80</td></tr>
<tr><td>0.75</td><td>0.17</td><td>51.23</td><td>83.42</td></tr>
<tr><td>0.7</td><td>0.57</td><td>65.52</td><td>88.77</td></tr>
<tr><td rowspan="3">ResNet-56</td><td>0.75</td><td>0.09</td><td>32.82</td><td>12.85</td></tr>
<tr><td>0.7</td><td>0.38</td><td>41.96</td><td>18.93</td></tr>
<tr><td>0.65</td><td>0.84</td><td>57.05</td><td>33.60</td></tr>
<tr><td rowspan="3">ResNet-110</td><td>0.75</td><td>0.02</td><td>35.24</td><td>8.81</td></tr>
<tr><td>0.7</td><td>0.44</td><td>45.54</td><td>15.32</td></tr>
<tr><td>0.65</td><td>1.67</td><td>62.67</td><td>32.11</td></tr>
<tr><td rowspan="3">DenseNet-40</td><td>0.75</td><td>-0.10</td><td>36.75</td><td>19.80</td></tr>
<tr><td>0.7</td><td>0.01</td><td>49.39</td><td>35.72</td></tr>
<tr><td>0.65</td><td>0.94</td><td>74.51</td><td>66.41</td></tr>
</table>

### [CIFAR-100](https://drive.google.com/drive/folders/1RabiD9TpQOBM00EZo1knBErRXR-cMRo4?usp=sharing)

#### VGGNet-16-BN

| Threshold | Top-1 Acc. Loss (%) | FR (%) | PR (%) |
|-----------|---------------------|--------|--------|
| 0.8       | -0.17               | 23.68  | 50.01  |
| 0.75      | 0.09                | 39.55  | 65.94  |
| 0.7       | 1.74                | 62.46  | 79.90  |

#### ResNet-56

| Threshold | Top-1 Acc. Loss (%) | FR (%) | PR (%) |
|-----------|---------------------|--------|--------|
| 0.75      | 0.22                | 29.24  | 7.77   |
| 0.7       | 0.71                | 47.02  | 17.50  |
| 0.65      | 2.10                | 60.32  | 33.01  |

#### ResNet-110

| Threshold | Top-1 Acc. Loss (%) | FR (%) | PR (%) |
|-----------|---------------------|--------|--------|
| 0.75      | 0.23                | 38.93  | 12.00  |
| 0.7       | 0.61                | 52.42  | 20.72|
| 0.65      | 1.95                | 67.08  | 46.78|

#### DenseNet-40

| Threshold | Top-1 Acc. Loss (%) | FR (%) | PR (%) |
|-----------|---------------------|--------|--------|
| 0.75      | 0.12                | 23.51  | 6.83   |
| 0.7       | 0.52                | 41.87  | 18.22  |
| 0.65      | 1.95                | 67.08  | 46.78  |

### [ImageNet](https://drive.google.com/drive/folders/1R84NTNa5UrZUImCPbdxVJovpcWWSDNpG?usp=sharing)

#### ResNet-50

| Threshold | Top-1 Acc. Loss (%) | Top-5 Acc. Loss (%) | FR (%) | PR (%) |
|-----------|---------------------|---------------------|--------|--------|
| 0.7       | 1.47                | 0.83                | 48.81  | 23.71  |
| 0.65      | 5.38                | 3.13                | 73.12  | 49.48  |

### [CUB-200](https://drive.google.com/drive/folders/16KbrS9UGWzhzTX1HdWq9ZDWptaIvoypm?usp=sharing)

| Model        | Threshold | Top-1 Acc. Loss (%) | FR (%) | PR (%) |
|--------------|-----------|---------------------|--------|--------|
| VGGNet-16-BN | 0.85      | 2.29                | 51.95  | 2.25   |
| VGGNet-16-BN | 0.85      | 1.85                | 52.66  | 2.91  |

#### VGGNet-19-BN
