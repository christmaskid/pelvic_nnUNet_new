
# list of datasets as "input_path|dataset_num|dataset_name"
if [[ "$1" == "1" ]]; then
    datasets=("1" "3" "7")
elif [[ "$1" == "2" ]]; then
    datasets=("5" "6") # later when we have the predictions from dataset 001 results for these datasets
else
    exit 1
fi

RESENC_SIZE="L" # "M"
LOGS=logs
mkdir -p LOGS

DIV_PART=5 # folds
gpu_nos=("5" "5" "6" "6" "7")

cmd="nnUNetv2_plan_experiment -d ${datasets[*]} -pl nnUNetPlannerResEnc${RESENC_SIZE}"
echo "Running: $cmd"
eval $cmd

    
for ds_num in ${datasets[@]}; do
    pids=()
    for DIV_ID in $(seq 0 $(($DIV_PART-1))); do
        logs="$LOGS/nnUNetv2_train_resenc-00${ds_num}_${DIV_ID}.log"
        cmd="CUDA_VISIBLE_DEVICES=${gpu_nos[$DIV_ID]} nnUNetv2_train \
            $ds_num 3d_fullres $DIV_ID \
            -p nnUNetResEncUNet${RESENC_SIZE}Plans > $logs 2>&1 &"
        echo "Running: $cmd"
        eval $cmd
        pids+=($!)
    done
    for pid in ${pids[@]}; do
        wait $pid
    done
done
