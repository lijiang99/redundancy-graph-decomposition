import os
import argparse
import platform
import torch
import torch.nn as nn
from small_scale.models import vgg16, densenet40, googlenet, mobilenet_v1, mobilenet_v2
from small_scale.models import resnet20, resnet32, resnet44, resnet56, resnet110
from small_scale.pruning import prune_vggnet_weights, prune_resnet_weights, prune_densenet_weights 
from small_scale.pruning import prune_googlenet_weights, prune_mobilenet_v1_weights, prune_mobilenet_v2_weights
from utils.data import load_cifar10, load_cifar100
from utils.calculate import train_on_others, validate_on_others
from thop import profile
from datetime import datetime
import json
import logging

parser = argparse.ArgumentParser(description="Fine-tune Pruned Model on CIFAR-10/100")

parser.add_argument("--root", type=str, default="./", help="project root directory")
parser.add_argument("--arch", type=str, default="vgg16", help="model architecture")
parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
parser.add_argument("--threshold", type=float, default=0.7, help="similarity threshold")
parser.add_argument("--epochs", type=int, default=70, help="number of fine-tuning epochs")
parser.add_argument("--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--learning-rate", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight-decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--step-size", type=int, default=30, help="learning rate decay step size")

def main():
    args = parser.parse_args()
    pruned_model_str = f"{args.arch}-{args.threshold}"
    
    # set for log file
    log_dir = os.path.join(args.root, args.dataset, "log", "fine-tune")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, f"{pruned_model_str}.log")
    if os.path.isfile(log_path):
        os.remove(log_path)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="a")
    sh = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    logger.info(f"author: jiang li - task: fine-tune pruned {args.arch} (threshold={args.threshold}) on {args.dataset}")
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
    
    # load pre-trained weights and model
    num_classes = 10 if args.dataset == "cifar10" else 100
    pretrain_weights_path = os.path.join(args.root, args.dataset, "pre-train", f"{args.arch}-weights.pth")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading weights from '{pretrain_weights_path}'")
    origin_model = eval(args.arch)(num_classes=num_classes).to(device)
    origin_state_dict = torch.load(pretrain_weights_path, map_location=device)
    origin_model.load_state_dict(origin_state_dict)
    dataset_dir = os.path.join(args.root, args.dataset, "dataset")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading dataset from '{dataset_dir}'")
    train_loader, val_loader = eval("load_"+args.dataset)(dataset_dir, batch_size=args.batch_size)
    
    # load pruning information
    prune_info_path = os.path.join(args.root, args.dataset, "prune-info", f"{pruned_model_str}.json")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading pruning information from '{prune_info_path}'")
    prune_info = None
    with open(prune_info_path, "r") as f:
        prune_info = json.load(f)
    
    # get name of conv layers and bn layers
    conv_layers, bn_layers, linear_layers = [], [], []
    for name, module in origin_model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(name)
        elif isinstance(module, nn.BatchNorm2d):
            bn_layers.append(name)
        elif isinstance(module, nn.Linear):
            linear_layers.append(name)
    
    # create pruned model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => creating pruned model '{pruned_model_str}'")
    pruned_model = eval(args.arch)(num_classes=num_classes, mask_nums=[value["mask_num"] for value in prune_info.values()]).to(device)
    pruned_state_dict = pruned_model.state_dict()
    logger.info(str(pruned_model))
    
    # load pruned weights to pruned model
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => loading pruned weights to pruned model '{pruned_model_str}'")
    if "vgg" in args.arch:
        pruned_state_dict = prune_vggnet_weights(prune_info=prune_info,
                                                 pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                 conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    elif "resnet" in args.arch:
        pruned_state_dict = prune_resnet_weights(prune_info=prune_info,
                                                 pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                 conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    elif "densenet" in args.arch:
        pruned_state_dict = prune_densenet_weights(prune_info=prune_info,
                                                   pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                   conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    elif "googlenet" in args.arch:
        pruned_state_dict = prune_googlenet_weights(prune_info=prune_info,
                                                    pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                    conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    elif "mobilenet_v1" == args.arch:
        pruned_state_dict = prune_mobilenet_v1_weights(prune_info=prune_info,
                                                       pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                       conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    elif "mobilenet_v2" == args.arch:
        pruned_state_dict = prune_mobilenet_v2_weights(prune_info=prune_info,
                                                       pruned_state_dict=pruned_state_dict, origin_state_dict=origin_state_dict,
                                                       conv_layers=conv_layers, bn_layers=bn_layers, linear_layers=linear_layers)
    
    pruned_model.load_state_dict(pruned_state_dict)
    
    # set hyperparameters
    optimizer = torch.optim.SGD(pruned_model.parameters(), lr=args.learning_rate,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # set the save path of the pruned model
    fine_tune_dir = os.path.join(args.root, args.dataset, "fine-tune")
    if not os.path.isdir(fine_tune_dir):
        os.makedirs(fine_tune_dir)
    save_path = os.path.join(fine_tune_dir, f"{pruned_model_str}-weights.pth")
    
    # start fine-tuning
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => fine-tuning pruned model '{pruned_model_str}'")
    best_acc = 0
    for epoch in range(args.epochs):
        beg_time = datetime.now()
        train_loss, train_acc = train_on_others(train_loader, pruned_model, criterion, optimizer, device)
        end_time = datetime.now()
        lr = optimizer.param_groups[0]["lr"]
        consume_time = int((end_time-beg_time).total_seconds())
        train_message = f"Epoch[{epoch+1:0>2}/{args.epochs}] - time: {consume_time:0>2}s - lr: {lr} - loss: {train_loss:.2f} - prec@1: {train_acc:.2f}"
        logger.info(train_message)
        valid_loss, valid_acc = validate_on_others(val_loader, pruned_model, criterion, device)
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(pruned_model.state_dict(), save_path)
        logger.info(f"Test - acc@1: {valid_acc:.2f} - best acc: {best_acc:.2f}")
        scheduler.step()
    
    # evaluate pruning effect
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => evaluating pruned model '{pruned_model_str}'")
    origin_best_acc, pruned_best_acc = validate_on_others(val_loader, origin_model, criterion, device)[1], best_acc
    input_image_size = 32
    input_image = torch.randn(1, 3, input_image_size, input_image_size).to(device)
    origin_flops, origin_params = profile(origin_model, inputs=(input_image,))
    pruned_flops, pruned_params = profile(pruned_model, inputs=(input_image,))
    logger.info(f"{'acc':<6}: {origin_best_acc:>6.2f}% -> {pruned_best_acc:>6.2f}% - drop: {origin_best_acc-pruned_best_acc:>5.2f}%")
    if args.arch == "googlenet":
        logger.info(f"{'flops':<6}: {origin_flops/1e9:>6.2f}G -> {pruned_flops/1e9:>6.2f}G - drop: {(origin_flops-pruned_flops)/origin_flops*100:>5.2f}%")
    else:
        logger.info(f"{'flops':<6}: {origin_flops/1e6:>6.2f}M -> {pruned_flops/1e6:>6.2f}M - drop: {(origin_flops-pruned_flops)/origin_flops*100:>5.2f}%")
    logger.info(f"{'params':<6}: {origin_params/1e6:>6.2f}M -> {pruned_params/1e6:>6.2f}M - drop: {(origin_params-pruned_params)/origin_params*100:>5.2f}%")
    logger.info(f"{datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | => done!")

if __name__ == "__main__":
    main()