# !/bin/bash
archs=("vgg16" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "densenet40" "googlenet" "mobilenet_v1" "mobilenet_v2")
pretrain_dir="./cifar100/pre-train/"
dataset_dir="./cifar100/dataset/"
prune_info_saved_dir="./cifar100/prune-info/"
prune_info_log_dir="./cifar100/log/generate_prune_info/"

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
	pretrain_path="./cifar100/pre-train/"$arch"-weights.pth"
	if [ ! -f "$pretrain_path" ]; then
		python train_cifar.py --arch $arch --dataset cifar100
	fi
	for threshold in ${thresholds[@]} 
	do
		prune_info_path="./cifar100/prune-info/"$arch"-"$threshold".json"
		if [ ! -f "$prune_info_path" ]; then
			python generate_prune_info.py --arch $arch --dataset cifar100 --pretrain-dir $pretrain_dir --dataset-dir $dataset_dir --saved-dir $prune_info_saved_dir --log-dir $prune_info_log_dir --threshold $threshold
		fi
		python prune_cifar.py --arch $arch --dataset cifar100 --threshold $threshold
	done
done
