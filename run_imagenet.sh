# !/bin/bash
archs=("resnet50" "vgg16_bn" "vgg19_bn")
dataset="imagenet"

for arch in ${archs[@]}
do
	case $arch in
		"vgg"*)
			thresholds=(0.7 0.75)
			;;
		"resnet50")
			thresholds=(0.65 0.7)
			;;
		*)
	esac
	for threshold in ${thresholds[@]}
	do
		prune_info_path=$dataset"/prune-info/"$arch"-"$threshold".json"
		if [ ! -f "$prune_info_path" ]; then
			python generate_prune_info.py --arch $arch --dataset $dataset --threshold $threshold
		fi
		python prune_imagenet.py --arch $arch --threshold $threshold
	done
done
