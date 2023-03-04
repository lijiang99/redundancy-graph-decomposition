import os
import argparse
import platform
import torch
import torch.nn as nn
from cifar10.models import vgg16, resnet56, resnet110, densenet40
from imagenet.models import resnet50
from cifar10.data import load_cifar10
from imagenet.data import load_imagenet
from utils.algorithm import FilterSelection
from datetime import datetime
import json
import logging

parser = argparse.ArgumentParser(description="Generate Pruning Information")

parser.add_argument("--arch", type=str, default="vgg16", help="model architecture")
parser.add_argument("--pretrain-dir", type=str, default="./cifar10/pre-train/", help="pre-trained model saved directory")
parser.add_argument("--dataset-dir", type=str, default="./cifar10/dataset/", help="dataset saved directory")
parser.add_argument("--saved-dir", type=str, default="./cifar10/prune-info/", help="pruning information saved directory")
parser.add_argument("--log-dir", type=str, default="./cifar10/log/generate-prune-info/", help="log file saved directory")
parser.add_argument("--mini-batch", type=int, default=100, help="number of inputs to calculate average similarity")
parser.add_argument("--threshold", type=float, default=0.7, help="similarity threshold")

feature_blobs = []
def feature_hook(module, inputs, output):
    """record output feature maps"""
    global feature_blobs
    feature_blobs.append(output.cpu().numpy())

def inference(train_loader, model, device, mini_batch=100):
    """inference the input of mini-batch"""
    inputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(train_loader):
            if i * images.shape[0] > mini_batch:
                break
            inputs.append(images)
        inputs = torch.cat(tuple(inputs))[:mini_batch]
        inputs = inputs.to(device)
        outputs = model(inputs)

def main():
    args = parser.parse_args()
    
    # set for log file
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    log_path = os.path.join(args.log_dir, f"{args.arch}-{args.threshold}.log")
    if os.path.isfile(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="a")
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    logger.info(f"author: jiang li - task: generate pruning information of {args.arch} (threshold={args.threshold})")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => printing arguments settings")
    args_info = str(args).replace(" ", "\n  ").replace("(", "(\n  ").replace(")", "\n)")
    logger.info(f"{args_info}")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => printing running environment")
    logger.info(f"{'python':<6} version: {platform.python_version()}")
    logger.info(f"{'torch':<6} version: {torch.__version__}")
    logger.info(f"{'cuda':<6} version: {torch.version.cuda}")
    logger.info(f"{'cudnn':<6} version: {torch.backends.cudnn.version()}")
    device = torch.device("cuda")
    device_prop = torch.cuda.get_device_properties(device)
    logger.info(f"{'device':<6} version: {device_prop.name} ({device_prop.total_memory/(1024**3):.2f} GB)")
    
    # create model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => creating model '{args.arch}'")
    model = eval(args.arch)().to(device)
    logger.info(str(model))
    
    # load weights and dataset
    pretrain_weights_path = os.path.join(args.pretrain_dir, f"{args.arch}-weights.pth")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading weights from '{pretrain_weights_path}'")
    state_dict = torch.load(pretrain_weights_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading dataset from '{args.dataset_dir}'")
    load_dataset = load_cifar10 if args.arch != "resnet50" else load_imagenet
    train_loader, val_loader = load_dataset(args.dataset_dir, batch_size=256)
    
    # inference to get output feature maps
    conv_layers, conv_weights = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
            conv_weights.append(state_dict[f"{name}.weight"])
            module.register_forward_hook(feature_hook)
    
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => extracting feature maps")
    inference(train_loader, model, device, mini_batch=args.mini_batch)
    
    # exclude the last convolutional layer of each residual structure for resnet
    sub_conv_layers, sub_conv_weights, sub_feature_blobs = [], [], []
    if args.arch in ["resnet56", "resnet110"]:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "conv2" not in name:
                sub_conv_layers.append(name)
    if args.arch == "resnet50":
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "conv3" not in name:
                sub_conv_layers.append(name)
    
    for conv_layer in sub_conv_layers:
        idx = conv_layers.index(conv_layer)
        sub_conv_weights.append(conv_weights[idx])
        sub_feature_blobs.append(feature_blobs[idx])
    
    # set the convolutional layer to be pruned
    final_conv_layers = sub_conv_layers if len(sub_conv_layers) != 0 else conv_layers
    final_conv_weights = sub_conv_weights if len(sub_conv_weights) != 0 else conv_weights
    final_feature_blobs = sub_feature_blobs if len(sub_feature_blobs) != 0 else feature_blobs
    
    # get the pruning information of the convolutional layers
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => selecting redundant filters")
    prune_info = dict()
    for conv_layer, feature_blob, weight in zip(final_conv_layers, final_feature_blobs, final_conv_weights):
        filter_selection = FilterSelection(feature_blob=feature_blob, weight=weight, threshold=args.threshold)
        beg_time = datetime.now()
        saved_filters = filter_selection.get_saved_filters()
        end_time = datetime.now()
        mask_num = weight.shape[0] - len(saved_filters)
        prune_info[conv_layer] = {"saved_idxs": saved_filters, "mask_num": mask_num}
        consume_time = str(end_time-beg_time).split('.')[0]
        logger.info(f"consume {consume_time}, remove {mask_num:0>4}/{weight.shape[0]:0>4} filters from '{conv_layer}'")
    
    # save information of pruning to json file
    save_path = os.path.join(args.saved_dir, f"{args.arch}-{args.threshold}.json")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => saving pruning information to '{save_path}'")
    if not os.path.isdir(args.saved_dir):
        os.makedirs(args.saved_dir)
    with open(save_path, "w") as f:
        json.dump(prune_info, f)
    
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => done!")

if __name__ == "__main__":
    main()