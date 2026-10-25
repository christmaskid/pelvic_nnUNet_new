nnUNet_raw="/data/b09401064/nnUNet_dataset/nnUNet_raw"
nnUNet_prediction="/data/b09401064/nnUNet_dataset/nnUNet_predictions"
mkdir -p $nnUNet_prediction

DIV_PART=3
gpu_nos=("0" "1" "2")
PLAN_NAME=${3:-"nnUNetPlans"} # nnUNetPlansResEncM, nnUNetPlansResEncL

# list of datasets as "train|test"
if [[ "$1" == "1" ]]; then
    test_cases=("gt|pred") # "global" "IPA" "gt|gt" 
elif [[ "$1" == "2" ]]; then
    test_cases=("pred|gt" "pred|pred" "mixed|gt" "mixed|pred")
else
    exit 1
fi

RUN_MODE=${2:-"all"} # prediction, metrics, all
if [[ $RUN_MODE == "prediction" ]]; then
    echo "Running prediction only"
    RUN_PREDICTION=true
    RUN_METRICS=false
elif [[ $RUN_MODE == "metrics" ]]; then
    echo "Running metrics only"
    RUN_PREDICTION=false
    RUN_METRICS=true
else
    RUN_PREDICTION=true
    RUN_METRICS=true
fi

declare -A mode_to_directory=( 
["gt"]="Dataset003_PelvicIPA" 
["pred"]="Dataset005_PelvicIPAPredInput" 
["mixed"]="Dataset006_PelvicIPAMixedInput" 
["global"]="Dataset001_Pelvic" 
["IPA"]="Dataset007_IPA"
)
image_dir="imagesTs"
declare -A mode_to_dsnum=(["gt"]="3" ["pred"]="5" ["mixed"]="6" ["global"]="1" ["IPA"]="7")

for test_case in ${test_cases[@]}; do
    echo $test_case

    if [[ "$test_case" == *"|"* ]]; then
        IFS='|' read -r TRAIN_MODE TEST_MODE <<< "$test_case"
        test_case="${TRAIN_MODE}_train_${TEST_MODE}_test"
    else
        TRAIN_MODE=$test_case
        TEST_MODE=$test_case
    fi
    
    INPUT_DIR="${nnUNet_raw}/${mode_to_directory[$TEST_MODE]}/${image_dir}"
    OUTPUT_DIR="${nnUNet_prediction}/${test_case}"
    mkdir -p "$OUTPUT_DIR"
    echo "$INPUT_DIR; $(ls -l $INPUT_DIR/*.nii.gz | wc -l)"
    echo $OUTPUT_DIR

    if [[ "$RUN_PREDICTION" == true ]]; then
        echo "Running prediction"
    
        pids=()
        for DIV_ID in $(seq 0 $(($DIV_PART-1))); do
            mkdir -p "$OUTPUT_DIR"
            log="$OUTPUT_DIR/nnUNetv2_predict-${test_case}-part${DIV_ID}.log"
            echo "Launching part $DIV_ID -> $log"
            cmd="CUDA_VISIBLE_DEVICES=$((${gpu_nos[$DIV_ID]})) nnUNetv2_predict \
                -i $INPUT_DIR -o $OUTPUT_DIR -p $PLAN_NAME \
                -d ${mode_to_dsnum[$TRAIN_MODE]} -c 3d_fullres -chk checkpoint_best.pth \
                -num_parts $DIV_PART -part_id $DIV_ID > $log 2>&1 &"
            echo "Running: $cmd"
            echo $cmd >> $log
            eval $cmd
            pids+=($!)
        done

        for pid in ${pids[@]}; do
            wait $pid
        done
    else
        echo "Skipping prediction"
    fi

    if [[ "$RUN_METRICS" == true ]]; then
        echo "Running metrics"

        # Testing metrics
        if [[ "$test_case" == "global" ]]; then
            GT_MODE="global"
        else
            GT_MODE="IPA"
        fi

        log="$OUTPUT_DIR/${test_case}-stats.log"
        cmd_test1="python test.py $OUTPUT_DIR 0 $GT_MODE >> $log 2>&1"
        cmd_test2="python test.py $OUTPUT_DIR 2 $GT_MODE >> $log 2>&1"
        echo "Running: $cmd_test1"
        eval $cmd_test1
        if [[ "$test_case" == *"IPA"* ]]; then
            echo "Running: $cmd_test2"
            eval $cmd_test2
        fi
    else
        echo "Skipping metrics"
    fi

done