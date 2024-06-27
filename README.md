# Efficient filter pruning: Reducing model complexity through redundancy graph decomposition ([Link](https://doi.org/10.1016/j.neucom.2024.128108))

## Requirements

| python |     torch    |  thop |  scipy | igraph |
|:------:|:------------:|:-----:|:------:|:------:|
| 3.8.10 | 1.11.0+cu113 | 0.1.1 | 1.10.1 | 0.11.5 |

## Running Code

### pre-training model

You can use `train.py` to train models from scratch on the CIFAR-10, CIFAR-100, and CUB-200 datasets for subsequent pruning experiments. Alternatively, you can directly employ the pretrained models provided [here](https://drive.google.com/drive/folders/1Y9x6OpJmK1nF0yzzIWHoa9JpnlxTXZkR?usp=sharing).

```
usage: train.py [-h] [--root ROOT] [--arch ARCH] [--dataset DATASET] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--step-size STEP_SIZE]

Train Model on CIFAR-10/100 or CUB-200 from Scratch

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           project root directory
  --arch ARCH           model architecture
  --dataset DATASET     dataset
  --epochs EPOCHS       number of training epochs
  --batch-size BATCH_SIZE
                        batch size
  --learning-rate LEARNING_RATE
                        initial learning rate
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY
                        weight decay
  --step-size STEP_SIZE
                        learning rate decay step size

# example: python train.py --arch resnet56 --dataset cifar10 --epochs 200 --step-size 50
```

### generating pruning information

You can use `generate_prune_info.py` to generate pruning information for any model based on specified thresholds. For the convenience of reproducible experiments, we provide the extracted pruning information for all models [here](https://drive.google.com/drive/folders/1MNzYwDrXdNzKiFgw2V2EXn_0OJ4GaMSx?usp=sharing).

```
usage: generate_prune_info.py [-h] [--root ROOT] [--arch ARCH] [--dataset DATASET] [--batch-size BATCH_SIZE] [--mini-batch MINI_BATCH] [--threshold THRESHOLD]

Generate Pruning Information

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           project root directory
  --arch ARCH           model architecture
  --dataset DATASET     dataset
  --batch-size BATCH_SIZE
                        batch size
  --mini-batch MINI_BATCH
                        number of inputs to calculate average similarity
  --threshold THRESHOLD
                        similarity threshold

# example: python generate_prune_info.py --arch resnet56 --dataset cifar10 --threshold 0.7
```

### pruning and fine-tuning model

You can prune and fine-tune a model on the CIFAR-10 and CIFAR-100 dataset by

```
usage: prune_cifar.py [-h] [--root ROOT] [--arch ARCH] [--dataset DATASET] [--threshold THRESHOLD] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--step-size STEP_SIZE]

Fine-tune Pruned Model on CIFAR-10/100

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           project root directory
  --arch ARCH           model architecture
  --dataset DATASET     dataset
  --threshold THRESHOLD
                        similarity threshold
  --epochs EPOCHS       number of fine-tuning epochs
  --batch-size BATCH_SIZE
                        batch size
  --learning-rate LEARNING_RATE
                        initial learning rate
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY
                        weight decay
  --step-size STEP_SIZE
                        learning rate decay step size

# example: python prune_cifar.py --arch resnet56 --dataset cifar10 --threshold 0.7
```

You can prune and fine-tune a model on the ImageNet (LSVRC-2012) dataset by

```
usage: prune_imagenet.py [-h] [--root ROOT] [--arch ARCH] [--dataset DATASET] [--threshold THRESHOLD] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--step-size STEP_SIZE]

Fine-tune Pruned Model on ImageNet

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           project root directory
  --arch ARCH           model architecture
  --dataset DATASET     dataset
  --threshold THRESHOLD
                        similarity threshold
  --epochs EPOCHS       number of fine-tuning epochs
  --batch-size BATCH_SIZE
                        batch size
  --learning-rate LEARNING_RATE
                        initial learning rate
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY
                        weight decay
  --step-size STEP_SIZE
                        learning rate decay step size

# example: python prune_imagenet.py --arch resnet50 --threshold 0.7
```

You can prune and fine-tune a model on the CUB-200 dataset by

```
usage: prune_cub200.py [-h] [--root ROOT] [--arch ARCH] [--dataset DATASET] [--threshold THRESHOLD] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY] [--step-size STEP_SIZE]

Fine-tune Pruned Model on CUB-200

optional arguments:
  -h, --help            show this help message and exit
  --root ROOT           project root directory
  --arch ARCH           model architecture
  --dataset DATASET     dataset
  --threshold THRESHOLD
                        similarity threshold
  --epochs EPOCHS       number of fine-tuning epochs
  --batch-size BATCH_SIZE
                        batch size
  --learning-rate LEARNING_RATE
                        initial learning rate
  --momentum MOMENTUM   momentum
  --weight-decay WEIGHT_DECAY
                        weight decay
  --step-size STEP_SIZE
                        learning rate decay step size

# example: python prune_cub200.py --arch vgg16_bn --threshold 0.85
```

### completing all experiments

You can complete all the aforementioned experiments at once by

```bash
./run.sh cifar10   # for all models on CIFAR-10
./run.sh cifar100  # for all models on CIFAR-100
./run.sh cub200   # for all models on CUB-200
./run.sh imagenet # for all models on ImageNet
```

## Experimental Results

We provide pruned models, along with comprehensive fine-tuning logs, to verify and reproduce the experimental results.

### [CIFAR-10](https://drive.google.com/drive/folders/1l-uM7ly4BJ_bF3w7QcVTSiqkiFu73Iu9?usp=sharing)

<table>
<tr align="center"><td>Model</td><td>Threshold</td><td>Top-1 Acc. Loss (%)</td><td>FR (%)</td><td>PR (%)</td></tr>
<tr align="center"><td rowspan="3">VGGNet-16-BN</td><td>0.8</td><td>-0.14</td><td>42.14</td><td>76.80</td></tr>
<tr align="center"><td>0.75</td><td>0.17</td><td>51.23</td><td>83.42</td></tr>
<tr align="center"><td>0.7</td><td>0.57</td><td>65.52</td><td>88.77</td></tr>
<tr align="center"><td rowspan="3">ResNet-56</td><td>0.75</td><td>0.09</td><td>32.82</td><td>12.85</td></tr>
<tr align="center"><td>0.7</td><td>0.38</td><td>41.96</td><td>18.93</td></tr>
<tr align="center"><td>0.65</td><td>0.84</td><td>57.05</td><td>33.60</td></tr>
<tr align="center"><td rowspan="3">ResNet-110</td><td>0.75</td><td>0.02</td><td>35.24</td><td>8.81</td></tr>
<tr align="center"><td>0.7</td><td>0.44</td><td>45.54</td><td>15.32</td></tr>
<tr align="center"><td>0.65</td><td>1.67</td><td>62.67</td><td>32.11</td></tr>
<tr align="center"><td rowspan="3">DenseNet-40</td><td>0.75</td><td>-0.10</td><td>36.75</td><td>19.80</td></tr>
<tr align="center"><td>0.7</td><td>0.01</td><td>49.39</td><td>35.72</td></tr>
<tr align="center"><td>0.65</td><td>0.94</td><td>74.51</td><td>66.41</td></tr>
</table>

### [CIFAR-100](https://drive.google.com/drive/folders/1RabiD9TpQOBM00EZo1knBErRXR-cMRo4?usp=sharing)

<table>
<tr align="center"><td>Model</td><td>Threshold</td><td>Top-1 Acc. Loss (%)</td><td>FR (%)</td><td>PR (%)</td></tr>
<tr align="center"><td rowspan="3">VGGNet-16-BN</td><td>0.8</td><td>-0.17</td><td>23.68</td><td>50.01</td></tr>
<tr align="center"><td>0.75</td><td>0.09</td><td>39.55</td><td>65.94</td></tr>
<tr align="center"><td>0.7</td><td>1.74</td><td>62.46</td><td>79.90</td></tr>
<tr align="center"><td rowspan="3">ResNet-56</td><td>0.75</td><td>0.22</td><td>29.24</td><td>7.77</td></tr>
<tr align="center"><td>0.7</td><td>0.71</td><td>47.02</td><td>17.50</td></tr>
<tr align="center"><td>0.65</td><td>2.10</td><td>60.32</td><td>33.01</td></tr>
<tr align="center"><td rowspan="3">ResNet-110</td><td>0.75</td><td>0.23</td><td>38.93</td><td>12.00</td></tr>
<tr align="center"><td>0.7</td><td>0.61</td><td>52.42</td><td>20.72</td></tr>
<tr align="center"><td>0.65</td><td>1.95</td><td>67.08</td><td>46.78</td></tr>
<tr align="center"><td rowspan="3">DenseNet-40</td><td>0.75</td><td>0.12</td><td>23.51</td><td>6.83</td></tr>
<tr align="center"><td>0.7</td><td>0.52</td><td>41.87</td><td>18.22</td></tr>
<tr align="center"><td>0.65</td><td>1.95</td><td>67.08</td><td>46.78</td></tr>
</table>

### [ImageNet](https://drive.google.com/drive/folders/1R84NTNa5UrZUImCPbdxVJovpcWWSDNpG?usp=sharing)

<table>
<tr align="center"><td>Model</td><td>Threshold</td><td>Top-1 Acc. Loss (%)</td><td>Top-5 Acc. Loss (%)</td><td>FR (%)</td><td>PR (%)</td></tr>
<tr align="center"><td rowspan="2">ResNet-50</td><td>0.7</td><td>1.47</td><td>0.83</td><td>48.81</td><td>23.71</td></tr>
<tr align="center"><td>0.65</td><td>5.38</td><td>3.13</td><td>73.12</td><td>49.48</td></tr>
</table>

### [CUB-200](https://drive.google.com/drive/folders/16KbrS9UGWzhzTX1HdWq9ZDWptaIvoypm?usp=sharing)

<table>
<tr align="center"><td>Model</td><td>Threshold</td><td>Top-1 Acc. Loss (%)</td><td>FR (%)</td><td>PR (%)</td></tr>
<tr align="center"><td>VGGNet-16-BN</td><td>0.85</td><td>2.29</td><td>51.95</td><td>2.25</td></tr>
<tr align="center"><td>VGGNet-16-BN</td><td>0.85</td><td>1.85</td><td>52.66</td><td>2.91</td></tr>
</table>

## Citation

If you find our pruning method useful in your research, please consider citing:

```
@article{li2024efficient,
  title={Efficient filter pruning: Reducing model complexity through redundancy graph decomposition},
  author={Li, Jiang and Shao, Haijian and Deng, Xing and Jiang, Yingtao},
  journal={Neurocomputing},
  pages={128108},
  year={2024},
  publisher={Elsevier}
}
```
