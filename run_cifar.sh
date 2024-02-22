# !/bin/bash
archs=("vgg16" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "densenet40" "googlenet" "mobilenet_v1" "mobilenet_v2")
dataset="cifar10"

for arch in ${archs[@]}
do
	case $arch in
		"vgg16")
			thresholds=(0.7 0.75 0.8)
			;;
		"resnet"*)
			thresholds=(0.65 0.7 0.75)
			;;
		"densenet40")
			thresholds=(0.65 0.7 0.75)
			;;
		"googlenet")
			thresholds=(0.7 0.75 0.8)
			;;
		"mobilenet"*)
			thresholds=(0.75 0.8)
			;;
		*)
	esac
	pretrain_path=$dataset"/pre-train/"$arch"-weights.pth"
	if [ ! -f "$pretrain_path" ]; then
		python train_cifar.py --arch $arch --dataset $dataset
	fi
	for threshold in ${thresholds[@]} 
	do
		prune_info_path=$dataset"/prune-info/"$arch"-"$threshold".json"
		if [ ! -f "$prune_info_path" ]; then
			python generate_prune_info.py --arch $arch --dataset $dataset --threshold $threshold
		fi
		python prune_cifar.py --arch $arch --dataset $dataset --threshold $threshold
	done
done
