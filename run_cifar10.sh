# !/bin/bash
archs=("vgg16" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "densenet40" "googlenet")

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
		*)
	esac
	for threshold in ${thresholds[@]} 
	do
		pretrain_path="./cifar10/pre-train/"$arch"-weights.pth"
		if [ ! -f "$pretrain_path" ]; then
			python train_cifar10.py --arch $arch
		fi
		prune_info_path="./cifar10/prune-info/"$arch"-"$threshold".json"
		if [ ! -f "$prune_info_path" ]; then
			python generate_prune_info.py --arch $arch --threshold $threshold
		fi
		python prune_cifar10.py --arch $arch --threshold $threshold
	done
done
