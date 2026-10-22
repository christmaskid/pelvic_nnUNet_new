nnUNet_raw="/data/b09401064/nnUNet_dataset/nnUNet_raw"
nnUNet_preprocessed="/data/b09401064/nnUNet_dataset/nnUNet_preprocessed"
nnUNet_results="/data/b09401064/nnUNet_dataset/nnUNet_results"

if [[ "$1" == "1" ]]; then
    datasets=("1" "3" "7")
elif [[ "$1" == "2" ]]; then
    datasets=("5" "6") # later when we have the predictions from dataset 001 results for these datasets
else
    exit 1
fi

DIV_PART=5 # folds
task_name="plain"
gpu_nos=("0" "1" "1" "2" "2")

mkdir -p logs

for ds_num in ${datasets[@]}; do
    pids=()
    for DIV_ID in $(seq 0 $(($DIV_PART-1))); do
        logs="logs/nnUNetv2_train_${task_name}-00${ds_num}_${DIV_ID}.log"
        cmd="CUDA_VISIBLE_DEVICES=$((${gpu_nos[$DIV_ID]})) \
            nnUNetv2_train $ds_num 3d_fullres $DIV_ID  > $logs 2>&1 &"
        echo "Running: $cmd"
        eval $cmd
        pids+=($!)
    done

    echo ${pids[@]}
    for pid in ${pids[@]}; do
        wait $pid
    done
done

## Remove processed data to save space
# for ds_num in ${datasets[@]}; do
#     cmd="rm ${nnUNet_preprocessed}/Dataset00${ds_num}* -rf"
#     echo "Running: $cmd"
#     eval $cmd
# done