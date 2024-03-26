# !/bin/bash

dataset=$1
if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
	archs=("vgg16_bn" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "densenet40" "googlenet" "mobilenet_v1" "mobilenet_v2")
elif [ "$dataset" == "cub200" ]; then
	archs=("vgg16_bn" "vgg19_bn")
elif [ "$dataset" == "imagenet" ]; then
	archs=("resnet50" "vgg16_bn" "vgg19_bn")
else
	echo "error: unexpected dataset"
	echo "optional datasets: ciafr10, cifar100, cub200 and imagenet"
	exit
fi

for arch in ${archs[@]}
do
	case $arch in
		"vgg"*)
			if [ "$dataset" == "cub200" ] || [ "$dataset" == "imagenet" ]; then
				thresholds=(0.8 0.85)
			else
				thresholds=(0.7 0.75 0.8)
			fi
			;;
		"resnet"*)
			if [ "$dataset" == "imagenet" ]; then
				thresholds=(0.65 0.7)
			else
				thresholds=(0.65 0.7 0.75)
			fi
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

	if [ "$dataset" != "imagenet" ]; then
		pretrain_path=$dataset"/pre-train/"$arch"-weights.pth"
		if [ ! -f "$pretrain_path" ]; then
			if [ "$dataset" == "cub200" ]; then
				python train.py --arch $arch --dataset $dataset --epochs 50 --batch-size 32 --learning-rate 0.01 --weight-decay 1e-5 --step-size 25
			else
				python train.py --arch $arch --dataset $dataset
			fi
		fi
	fi

	for threshold in ${thresholds[@]}
	do
		prune_info_path=$dataset"/prune-info/"$arch"-"$threshold".json"
		if [ ! -f "$prune_info_path" ]; then
			python generate_prune_info.py --arch $arch --dataset $dataset --threshold $threshold
		fi
		case $dataset in
			"cifar"*)
				exec_file=prune_cifar.py
				;;
			"cub200")
				exec_file=prune_cub200.py
				;;
			"imagenet")
				exec_file=prune_imagenet.py
				;;
			*)
		esac
		python $exec_file --arch $arch --dataset $dataset --threshold $threshold
	done
done
