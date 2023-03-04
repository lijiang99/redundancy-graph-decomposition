# Filter Pruning via Decomposition of Redundancy Graphs

## Requirements

| Python | PyTorch | CUDA |
|--------|---------|------|
| 3.7.0  | 1.7.1   | 11.0 |

## Generate pruning information

You can set the threshold to generate pruning information for the model by
```bash
usage: generate_prune_info.py [-h] [--arch ARCH] [--pretrain-dir PRETRAIN_DIR]
                              [--dataset-dir DATASET_DIR]
                              [--saved-dir SAVED_DIR] [--log-dir LOG_DIR]
                              [--mini-batch MINI_BATCH]
                              [--threshold THRESHOLD]

Generate Pruning Information

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           model architecture
  --pretrain-dir PRETRAIN_DIR
                        pre-trained model saved directory
  --dataset-dir DATASET_DIR
                        dataset saved directory
  --saved-dir SAVED_DIR
                        pruning information saved directory
  --log-dir LOG_DIR     log file saved directory
  --mini-batch MINI_BATCH
                        number of inputs to calculate average similarity
  --threshold THRESHOLD
                        similarity threshold
```
For the convenience of reproducible experiments, we provide the extracted pruning information for all models [here](https://drive.google.com/drive/folders/1UogjNSnSxqxbtmA5tZBrJJOlU11t_R3e?usp=share_link).

## Fine-tuning the pruned model

You can prune and fine-tune a model on the CIFAR-10 dataset by
```bash
usage: prune_cifar10.py [-h] [--arch ARCH] [--pretrain-dir PRETRAIN_DIR]
                        [--dataset-dir DATASET_DIR]
                        [--pruneinfo-dir PRUNEINFO_DIR]
                        [--saved-dir SAVED_DIR] [--log-dir LOG_DIR]
                        [--threshold THRESHOLD] [--epochs EPOCHS]
                        [--batch-size BATCH_SIZE]
                        [--learning-rate LEARNING_RATE] [--momentum MOMENTUM]
                        [--weight-decay WEIGHT_DECAY] [--adjust ADJUST]

Fine-tune Pruned Model on CIFAR-10

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           model architecture
  --pretrain-dir PRETRAIN_DIR
                        pre-trained model saved directory
  --dataset-dir DATASET_DIR
                        dataset saved directory
  --pruneinfo-dir PRUNEINFO_DIR
                        pruning information saved directory
  --saved-dir SAVED_DIR
                        pruned model saved directory
  --log-dir LOG_DIR     log file saved directory
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
  --adjust ADJUST       learning rate decay step
```

You can prune and fine-tune a model on the ImageNet (LSVRC-2012) dataset by
```bash
usage: prune_imagenet.py [-h] [--arch ARCH] [--pretrain-dir PRETRAIN_DIR]
                         [--dataset-dir DATASET_DIR]
                         [--pruneinfo-dir PRUNEINFO_DIR]
                         [--saved-dir SAVED_DIR] [--log-dir LOG_DIR]
                         [--threshold THRESHOLD] [--epochs EPOCHS]
                         [--batch-size BATCH_SIZE]
                         [--learning-rate LEARNING_RATE] [--momentum MOMENTUM]
                         [--weight-decay WEIGHT_DECAY] [--adjust ADJUST]

Fine-tune Pruned Model on ImageNet

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH           model architecture
  --pretrain-dir PRETRAIN_DIR
                        pre-trained model saved directory
  --dataset-dir DATASET_DIR
                        dataset saved directory
  --pruneinfo-dir PRUNEINFO_DIR
                        pruning information saved directory
  --saved-dir SAVED_DIR
                        pruned model saved directory
  --log-dir LOG_DIR     log file saved directory
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
  --adjust ADJUST       learning rate decay step
```

## Examples

You need to prepare the CIFAR-10 and ImageNet (LSVRC-2012) datasets, and download the pre-trained models [here](https://drive.google.com/drive/folders/1jmQwvBw8i_HgWKR4-GqwZpc49jLZgLx7?usp=share_link) for pruning. You can also check the [pruned model](https://drive.google.com/drive/folders/1n--QtCW1STOp8o890YLvlZzNm5WLZOiv?usp=share_link) and fine-tuning [log files](https://drive.google.com/drive/folders/1v86bkBgO6LPef_5oCjCKj9yCLYvhH6-c?usp=share_link). When running examples, if optional parameters are not set, default values are used.

### VGG-16

| threshold | Top-1  | Top-1 drop | FLOPs drop | Params drop |
|-----------|--------|------------|------------|-------------|
| baseline  | 93.72% | 0%         | 0%         | 0%          |
| 0.8       | 93.56% | 0.16%      | 42.16%     | 77.11%      |
| 0.75      | 93.37% | 0.35%      | 51.24%     | 83.58%      |
| 0.7       | 92.93% | 0.79%      | 65.52%     | 88.88%      |

```bash
python generate_prune_info.py --arch vgg16 --threshold [0.7 0.75 0.8]
python prune_cifar10.py --arch vgg16 --threshold [0.7 0.75 0.8]
```

### ResNet-56

| threshold | Top-1  | Top-1 drop | FLOPs drop | Params drop |
|-----------|--------|------------|------------|-------------|
| baseline  | 93.45% | 0%         | 0%         | 0%          |
| 0.75      | 93.18% | 0.27%      | 32.82%     | 12.85%      |
| 0.7       | 93.07% | 0.38%      | 41.96%     | 18.93%      |
| 0.65      | 92.52% | 0.93%      | 57.05%     | 33.60%      |

```bash
python generate_prune_info.py --arch resnet56 --threshold [0.65 0.7 0.75]
python prune_cifar10.py --arch resnet56 --threshold [0.65 0.7 0.75]
```

### ResNet-110

| threshold | Top-1  | Top-1 drop | FLOPs drop | Params drop |
|-----------|--------|------------|------------|-------------|
| baseline  | 93.94% | 0%         | 0%         | 0%          |
| 0.75      | 93.86% | 0.08%      | 35.24%     | 8.81%       |
| 0.7       | 93.43% | 0.51%      | 45.54%     | 15.32%      |
| 0.65      | 92.25% | 1.69%      | 62.67%     | 32.11%      |

```bash
python generate_prune_info.py --arch resnet110 --threshold [0.65 0.7 0.75]
python prune_cifar10.py --arch resnet110 --threshold [0.65 0.7 0.75]
```

### DenseNet-40

| threshold | Top-1  | Top-1 drop | FLOPs drop | Params drop |
|-----------|--------|------------|------------|-------------|
| baseline  | 93.87% | 0%         | 0%         | 0%          |
| 0.75      | 93.92% | -0.05%     | 36.75%     | 19.80%      |
| 0.7       | 93.78% | 0.09%      | 49.39%     | 35.72%      |
| 0.65      | 92.76% | 1.11%      | 74.51%     | 66.41%      |

```bash
python generate_prune_info.py --arch densenet40 --threshold [0.65 0.7 0.75]
python prune_cifar10.py --arch densenet40 --threshold [0.65 0.7 0.75]
```

### ResNet-50

| threshold | Top-1  | Top-5  | Top-1 drop | Top-5 drop | FLOPs drop | Params drop |
|-----------|--------|--------|------------|------------|------------|-------------|
| baseline  | 76.15% | 92.87% | 0%         | 0%         | 0%         | 0%          |
| 0.7       | 74.61% | 91.93% | 1.54%      | 0.94%      | 48.81%     | 23.71%      |
| 0.65      | 69.90% | 89.21% | 6.25%      | 3.66%      | 73.12%     | 49.48%      |

```bash
python generate_prune_info.py --arch resnet50 --pretrain-dir ./imagenet/pre-train/ --dataset-dir ./imagenet/dataset/ --saved-dir ./imagenet/prune-info/ --log-dir ./imagenet/log/generate_prune_info/ --threshold [0.65 0.7]
python prune_imagenet.py --arch resnet50 --threshold [0.65 0.7]
```
### Tips

You can implement the complete experiment including generating pruning information and fine-tuning by
```bash
source run_cifar10.sh
source run_imagenet.sh
```
