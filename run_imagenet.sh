# !/bin/bash
archs=("resnet50" "vgg16_bn" "vgg19_bn")
pretrain_dir="./imagenet/pre-train/"
dataset_dir="./imagenet/dataset/"
prune_info_saved_dir="./imagenet/prune-info/"
prune_info_log_dir="./imagenet/log/generate_prune_info/"

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
		prune_info_path="./imagenet/prune-info/"$arch"-"$threshold".json"
		if [ ! -f "$prune_info_path" ]; then
			python generate_prune_info.py --arch $arch --pretrain-dir $pretrain_dir --dataset-dir $dataset_dir --saved-dir $prune_info_saved_dir --log-dir $prune_info_log_dir --threshold $threshold
		fi
		python prune_imagenet.py --arch $arch --threshold $threshold
	done
done
