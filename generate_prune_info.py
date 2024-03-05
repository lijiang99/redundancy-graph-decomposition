import os
import argparse
import torch
import torch.nn as nn
import torchvision
from small_scale.models import vgg16, densenet40, googlenet, mobilenet_v1, mobilenet_v2
from small_scale.models import resnet20, resnet32, resnet44, resnet56, resnet110
from large_scale.models import vgg16_bn, vgg19_bn, resnet50
from utils.data import load_cifar10, load_cifar100, load_cub200, load_imagenet
from utils.algorithm import FilterSelection
from utils.logger import Logger
from datetime import datetime
import json

parser = argparse.ArgumentParser(description="Generate Pruning Information")

parser.add_argument("--root", type=str, default="./", help="project root directory")
parser.add_argument("--arch", type=str, default="vgg16", help="model architecture")
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
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
    log_dir = os.path.join(args.root, args.dataset, "log", "generate-prune-info")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{args.arch}-{args.threshold}.log")
    logger = Logger(log_path)
    
    logger.mesg(f"author: jiang li - task: generate pruning information of {args.arch} (threshold={args.threshold})")
    logger.hint("printing arguments settings")
    logger.args(args)
    logger.hint("printing running environment")
    device = torch.device("cuda")
    logger.envs(device)
    
    # create model
    logger.hint(f"creating model '{args.arch}'")
    if args.dataset == "imagenet":
        model = eval(f"torchvision.models.{args.arch}")(pretrained=True).to(device)
    else:
        model = eval(args.arch)(num_classes=(10 if args.dataset == "cifar10" else (100 if args.dataset == "cifar100" else 200))).to(device)
    logger.mesg(str(model))
    
    # load pre-trained weights and dataset
    if args.dataset != "imagenet":
        pretrain_weights_path = os.path.join(args.root, args.dataset, "pre-train", f"{args.arch}-weights.pth")
        logger.hint(f"loading weights from '{pretrain_weights_path}'")
        model.load_state_dict(torch.load(pretrain_weights_path, map_location=device))
    
    state_dict = model.state_dict()
    dataset_dir = os.path.join(args.root, args.dataset, "dataset")
    logger.hint(f"loading dataset from '{dataset_dir}'")
    train_loader, val_loader = eval("load_"+args.dataset)(dataset_dir, batch_size=args.batch_size)
    
    # inference to get output feature maps
    conv_layers, conv_weights = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
            conv_weights.append(state_dict[f"{name}.weight"])
            module.register_forward_hook(feature_hook)
    
    logger.hint("extracting feature maps")
    inference(train_loader, model, device, mini_batch=args.mini_batch)
    
    # exclude the last convolutional layer of each residual structure for resnet
    sub_conv_layers, sub_conv_weights, sub_feature_blobs = [], [], []
    if args.arch in [f"resnet{depth}" for depth in [20,32,44,56,110]]:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "conv2" not in name:
                sub_conv_layers.append(name)
    if args.arch == "resnet50":
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and "conv3" not in name:
                sub_conv_layers.append(name)
    if args.arch == "mobilenet_v1":
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and (name == "model.0.0" or name.split(".")[2] == "3"):
                sub_conv_layers.append(name)
    if args.arch == "mobilenet_v2":
        for name, module in model.named_modules():
            if ((isinstance(module, nn.Conv2d)) and
                ((name in ["conv1", "conv2"]) or ("shortcut" in name) or ("conv1" in name)
                 or (("conv3" in name) and (name.split(".")[1] in ["3", "6", "13"])))):
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
    logger.hint("selecting redundant filters")
    prune_info = dict()
    for conv_layer, feature_blob, weight in zip(final_conv_layers, final_feature_blobs, final_conv_weights):
        beg_time = datetime.now()
        filter_selection = FilterSelection(feature_blob=feature_blob, weight=weight.cpu().numpy(), threshold=args.threshold)
        saved_filters = filter_selection.get_saved_filters()
        end_time = datetime.now()
        mask_num = weight.shape[0] - len(saved_filters)
        prune_info[conv_layer] = {"saved_idxs": saved_filters, "mask_num": mask_num}
        consume_time = str(end_time-beg_time).split('.')[0]
        logger.mesg(f"consume {consume_time}, remove {mask_num:0>4}/{weight.shape[0]:0>4} filters from '{conv_layer}'")
    
    # save information of pruning to json file
    save_dir = os.path.join(args.root, args.dataset, "prune-info")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{args.arch}-{args.threshold}.json")
    logger.hint(f"saving pruning information to '{save_path}'")
    
    with open(save_path, "w") as f:
        json.dump(prune_info, f)
    
    logger.hint("done!")

if __name__ == "__main__":
    main()