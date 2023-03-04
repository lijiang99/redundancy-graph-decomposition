import torch

def prune_vggnet_weights(prune_info, pruned_state_dict, origin_state_dict, conv_layers, bn_layers):
    """prune vggnet weights based on pruning information"""
    in_saved_idxs = [0,1,2]
    for conv_layer, bn_layer in zip(conv_layers, bn_layers):
        out_saved_idxs = prune_info[conv_layer]["saved_idxs"]
        pruned_conv_weight = origin_state_dict[f"{conv_layer}.weight"][out_saved_idxs,:,:,:][:,in_saved_idxs,:,:]
        pruned_state_dict[f"{conv_layer}.weight"] = pruned_conv_weight
        bn_params = ["bias", "running_mean", "running_var", "weight"]
        for bn_param in bn_params:
            pruned_bn_param = origin_state_dict[f"{bn_layer}.{bn_param}"][out_saved_idxs]
            pruned_state_dict[f"{bn_layer}.{bn_param}"] = pruned_bn_param
        in_saved_idxs = out_saved_idxs
    return pruned_state_dict

def prune_resnet_weights(prune_info, pruned_state_dict, origin_state_dict, conv_layers, bn_layers):
    """prune resnet weights based on pruning information"""
    in_saved_idxs, out_saved_idxs = [0,1,2], None
    last_downsample_out_saved_idxs = prune_info["conv"]["saved_idxs"]
    for conv_layer, bn_layer in zip(conv_layers, bn_layers):
        downsample_layer = "conv" if ("layer1" in conv_layer or conv_layer == "conv") else f"{conv_layer.split('.')[0]}.0.downsample.0"
        downsample_out_saved_idxs = prune_info[downsample_layer]["saved_idxs"]
        if "layer" in conv_layer and "conv2" in conv_layer:
            in_saved_idxs = prune_info[conv_layer.replace("conv2", "conv1")]["saved_idxs"]
            out_saved_idxs = downsample_out_saved_idxs
        elif "downsample" in conv_layer:
            in_saved_idxs = last_downsample_out_saved_idxs
            out_saved_idxs = prune_info[conv_layer]["saved_idxs"]
            last_downsample_out_saved_idxs = downsample_out_saved_idxs
        else:
            out_saved_idxs = prune_info[conv_layer]["saved_idxs"]
        pruned_conv_weight = origin_state_dict[f"{conv_layer}.weight"][out_saved_idxs,:,:,:][:,in_saved_idxs,:,:]
        pruned_state_dict[f"{conv_layer}.weight"] = pruned_conv_weight
        bn_params = ["bias", "running_mean", "running_var", "weight"]
        for bn_param in bn_params:
            pruned_bn_param = origin_state_dict[f"{bn_layer}.{bn_param}"][out_saved_idxs]
            pruned_state_dict[f"{bn_layer}.{bn_param}"] = pruned_bn_param
        in_saved_idxs = downsample_out_saved_idxs
    return pruned_state_dict

def prune_densenet_weights(prune_info, pruned_state_dict, origin_state_dict, conv_layers, bn_layers):
    """prune densenet weights based on pruning information"""
    in_saved_idxs, last_in_saved_idxs = [0,1,2], []
    for conv_layer, bn_layer in zip(conv_layers, bn_layers):
        out_saved_idxs = prune_info[conv_layer]["saved_idxs"]
        origin_conv_weight = origin_state_dict[f"{conv_layer}.weight"]
        pruned_conv_weight = origin_conv_weight[out_saved_idxs,:,:,:][:,in_saved_idxs,:,:]
        pruned_state_dict[f"{conv_layer}.weight"] = pruned_conv_weight
        if conv_layer == "conv" or "trans" in conv_layer:
            in_saved_idxs = out_saved_idxs
            last_in_saved_idxs = out_saved_idxs
        else:
            offset = list((torch.tensor(out_saved_idxs)+origin_conv_weight.shape[1]).cpu().numpy())
            in_saved_idxs = last_in_saved_idxs + offset
            last_in_saved_idxs = in_saved_idxs
        bn_params = ["bias", "running_mean", "running_var", "weight"]
        for bn_param in bn_params:
            pruned_bn_param = origin_state_dict[f"{bn_layer}.{bn_param}"][in_saved_idxs]
            pruned_state_dict[f"{bn_layer}.{bn_param}"] = pruned_bn_param
    return pruned_state_dict