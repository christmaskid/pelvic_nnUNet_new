## PREPROCESSING
datasets=("1" "3" "5" "6" "7")
task_name="plain"
mkdir -p logs

for ds_num in ${datasets[@]}; do
    log=logs/preprocess_${task_name}-00${ds_num}.log
    echo "Processing dataset $ds_num"
    cmd="nnUNetv2_plan_and_preprocess -d $ds_num -c 3d_fullres \
    --verify_dataset_integrity > $log 2>&1"
    echo "Running: $cmd"
    eval $cmd
done