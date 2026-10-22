if [[ "$1" == "1" ]]; then
    datasets=("1" "3" "7")
elif [[ "$1" == "2" ]]; then
    datasets=("5" "6") # later when we have the predictions from dataset 001 results for these datasets
fi
task_name="plain"
mkdir -p logs

for ds_num in ${datasets[@]}; do
    log=logs/preprocess_${task_name}-00${ds_num}.log
    echo "Processing dataset $ds_num"
    cmd="nnUNetv2_plan_and_preprocess -d $ds_num -c 3d_fullres \
    --verify_dataset_integrity > $log 2>&1" # Don't make it parallel because of RAM issue
    echo "Running: $cmd"
    eval $cmd
done