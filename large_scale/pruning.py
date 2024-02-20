def prune_vggnet_weights(prune_info, pruned_state_dict, origin_state_dict, conv_layers, bn_layers, linear_layers):
    """prune vggnet weights based on pruning information"""
    in_saved_idxs = [0,1,2]
    for conv_layer, bn_layer in zip(conv_layers, bn_layers):
        out_saved_idxs = prune_info[conv_layer]["saved_idxs"]
        pruned_conv_weight = origin_state_dict[f"{conv_layer}.weight"][out_saved_idxs,:,:,:][:,in_saved_idxs,:,:]
        pruned_conv_bias = origin_state_dict[f"{conv_layer}.bias"][out_saved_idxs]
        pruned_state_dict[f"{conv_layer}.weight"] = pruned_conv_weight
        pruned_state_dict[f"{conv_layer}.bias"] = pruned_conv_bias
        bn_params = ["bias", "running_mean", "running_var", "weight"]
        for bn_param in bn_params:
            pruned_bn_param = origin_state_dict[f"{bn_layer}.{bn_param}"][out_saved_idxs]
            pruned_state_dict[f"{bn_layer}.{bn_param}"] = pruned_bn_param
        in_saved_idxs = out_saved_idxs
    for i, linear_layer in enumerate(linear_layers):
        if i == 0:
            saved_idxs = []
            for in_saved_idx in in_saved_idxs:
                saved_idxs += list(range(in_saved_idx*7*7, in_saved_idx*7*7+7*7))
            pruned_state_dict[f"{linear_layer}.weight"] = origin_state_dict[f"{linear_layer}.weight"][:,saved_idxs]
        else:
            pruned_state_dict[f"{linear_layer}.weight"] = origin_state_dict[f"{linear_layer}.weight"]
        pruned_state_dict[f"{linear_layer}.bias"] = origin_state_dict[f"{linear_layer}.bias"]
    return pruned_state_dict

def prune_resnet_weights(prune_info, pruned_state_dict, origin_state_dict, conv_layers, bn_layers, linear_layers):
    """prune resnet weights based on pruning information"""
    in_saved_idxs, out_saved_idxs = [0,1,2], None
    last_downsample_out_saved_idxs = prune_info["conv1"]["saved_idxs"]
    for conv_layer, bn_layer in zip(conv_layers, bn_layers):
        downsample_layer = None if conv_layer == "conv1" else conv_layer.split(".")[0]+".0.downsample.0"
        downsample_out_saved_idxs = prune_info[downsample_layer]["saved_idxs"] if downsample_layer else []
        if "conv1" in conv_layer or "conv2" in conv_layer:
            out_saved_idxs = prune_info[conv_layer]["saved_idxs"]
        if "conv3" in conv_layer:
            out_saved_idxs = downsample_out_saved_idxs
        if "downsample" in conv_layer:
            in_saved_idxs = last_downsample_out_saved_idxs
            out_saved_idxs = downsample_out_saved_idxs
        
        pruned_conv_weight = origin_state_dict[conv_layer+".weight"][out_saved_idxs,:,:,:][:,in_saved_idxs,:,:]
        pruned_state_dict[conv_layer+".weight"] = pruned_conv_weight
        bn_params = ["bias", "running_mean", "running_var", "weight"]
        for bn_param in bn_params:
            pruned_bn_param = origin_state_dict[bn_layer+"."+bn_param][out_saved_idxs]
            pruned_state_dict[bn_layer+"."+bn_param] = pruned_bn_param
        
        if "conv1" in conv_layer or "conv2" in conv_layer:
            in_saved_idxs = out_saved_idxs
        if "conv3" in conv_layer:
            in_saved_idxs = downsample_out_saved_idxs
        if "downsample" in conv_layer:
            in_saved_idxs = downsample_out_saved_idxs
            last_downsample_out_saved_idxs = downsample_out_saved_idxs
    for i, linear_layer in enumerate(linear_layers):
        if i == 0:
            pruned_state_dict[f"{linear_layer}.weight"] = origin_state_dict[f"{linear_layer}.weight"][:,in_saved_idxs]
        else:
            pruned_state_dict[f"{linear_layer}.weight"] = origin_state_dict[f"{linear_layer}.weight"]
        pruned_state_dict[f"{linear_layer}.bias"] = origin_state_dict[f"{linear_layer}.bias"]
    return pruned_state_dict