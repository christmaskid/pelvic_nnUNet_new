nohup bash 00_link_and_rename_dataset.sh > logs/00_link_and_rename_dataset.log &
nohup python 01_convert_dataset.py > logs/01_convert_dataset.log &

---
nohup bash 02_preprocess.sh 1 > logs/02_preprocess.log &
  |-- nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres     --verify_dataset_integrity > logs/preprocess_plain-001.log 2>&1 &
  |-- nnUNetv2_plan_and_preprocess -d 3 -c 3d_fullres     --verify_dataset_integrity > logs/preprocess_plain-003.log 2>&1 &
  └-- nnUNetv2_plan_and_preprocess -d 7 -c 3d_fullres     --verify_dataset_integrity > logs/preprocess_plain-007.log 2>&1 &
nohup bash 03_train.sh 1 > logs/03_train.log &
nohup bash 05_pred.sh 1 > logs/05_pred.log &
nohup bash 04_test.sh 1 > logs/04_test.log & # gt_train_pred_test needs pred as input

nohup bash 02_preprocess.sh 2 >> logs/02_preprocess.log &
nohup bash 03_train.sh 2 >> logs/03_train.log &
nohup bash 04_test.sh 2 >> logs/04_test.log &

---
nohup bash 03-1_preprocess-and-train_resenc.sh 1 > logs/03-1_preprocess-and-train_resenc.log &