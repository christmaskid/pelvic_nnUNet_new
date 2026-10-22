# Stats
import numpy as np
from glob import glob
import nibabel as nib
import csv
import os
import sys
import pandas as pd
from scipy.ndimage import label as connected_label
from monai.metrics import compute_hausdorff_distance

from skimage.morphology import skeletonize_3d
from pathlib import Path
from my_skeletonize import skeletonize_to_voxels

def compute_dice(gt, pred, smooth = 1e-6):
    intersection = np.sum(gt * pred)
    return (2. * intersection + smooth) / (np.sum(gt) + np.sum(pred) + smooth)

def compute_precision(gt, pred):
    intersection = np.sum(gt * pred)
    return intersection / np.sum(pred)

def compute_recall(gt, pred):
    intersection = np.sum(gt * pred)
    return intersection / np.sum(gt)

def compute_iou(gt, pred, smooth=1e-6):
    intersection = np.sum(gt * pred)
    return (intersection + smooth) / (np.sum(gt) + np.sum(pred) - intersection + smooth)

DEFAULT_RAW_ROOT ="/data/b09401064/nnUNet_dataset/nnUNet_raw"
DEBUG = False # True # 
DEFAULT_SEGMENTATION_ROOT ="/data/b09401064/nnUNet_dataset/nnUNet_segmentations"
DEFAULT_TEST_IDS_PATH = "/home/b09401064/pelvic_CTA/nnUNet_code/csv/test.csv"

if __name__=="__main__":    
    assert len(sys.argv)>1, "Please provide dataset name as argument."
    test_dir = sys.argv[1]
    n_pick_components = eval(sys.argv[2]) if len(sys.argv)>2 else 2
    ground_truth_mode = sys.argv[3] if len(sys.argv)>3 else "IPA"
    test_mode = sys.argv[4] if len(sys.argv)>4 else "test"
    csv_name = sys.argv[5] if len(sys.argv)>5 else f"test_km_keep-{n_pick_components}_{ground_truth_mode}.csv"

    assigned_test_ids_path = sys.argv[6] if len(sys.argv)>6 else DEFAULT_TEST_IDS_PATH
    df = pd.read_csv(assigned_test_ids_path)
    assigned_test_ids = df['RENAMED_ID'].tolist()
    assigned_test_ids = [str(x).zfill(4) for x in assigned_test_ids]
    print("Assigned test IDs loaded from CSV:", assigned_test_ids, flush=True)

    # all_gts = sorted(glob(f"{DEFAULT_RAW_ROOT}/Dataset001_Pelvic/labelsTs/*.nii.gz"))
    # IPA_gts = sorted(glob(f"{DEFAULT_RAW_ROOT}/Dataset007_IPA/labelsTs/*.nii.gz"))
    
    all_gts_dir = f"{DEFAULT_RAW_ROOT}/Dataset001_Pelvic/labelsTs"
    all_gts = sorted([os.path.join(all_gts_dir, f+".nii.gz") for f in assigned_test_ids if os.path.exists(os.path.join(all_gts_dir, f+".nii.gz"))])
    IPA_gts_dir = f"{DEFAULT_RAW_ROOT}/Dataset007_IPA/labelsTs"
    IPA_gts = sorted([os.path.join(IPA_gts_dir, f+".nii.gz") for f in assigned_test_ids if os.path.exists(os.path.join(IPA_gts_dir, f+".nii.gz"))])

    gts_list, class_names_list = [], []
    if ground_truth_mode == "global":
        gts_list = [all_gts, IPA_gts]
        class_names_list = ["global", "IPA"]
    elif ground_truth_mode == "IPA":
        gts_list = [IPA_gts]
        class_names_list = ["IPA"]
    else:
        raise NotImplementedError("ground_truth_mode {} not implemented.".format(ground_truth_mode))
    preds = sorted(glob(os.path.join(test_dir, "*.nii.gz")))
    print(len(all_gts), len(preds), flush=True)

    postprocessing_mode = "none" 
    if n_pick_components>1:
        postprocessing_mode = "number"
    elif n_pick_components>0 and n_pick_components<1:
        postprocessing_mode = "percentage"

    test_fn = f"test_km_keep-{n_pick_components}.csv" if csv_name is None else csv_name

    if postprocessing_mode != "none":
        save_dir = os.path.join(test_dir, "keep-{}".format(n_pick_components))
        os.makedirs(save_dir, exist_ok=True)

    file = open(os.path.join(test_dir, test_fn), "w")
    # file = open(test_dir + "test_fold0.csv", "w")
    writer = csv.writer(file)
    writer.writerow(["fn", "class", "dice", "precision", "recall", "iou", 
                    "HD95", "ASD", "t_prec", "t_sens", "clDice"])

    for gts, class_name in zip(gts_list, class_names_list):
        for gt_fn, pred_fn in zip(gts, preds):
            
            fn = os.path.basename(gt_fn)[:-len(".nii.gz")]
            # if eval(re.sub(r'\b0+(?!\b)', '', fn.split("_")[2])) > 70:
            #     continue
            print(fn, flush=True)
            
            gt, pred = nib.load(gt_fn).get_fdata(), nib.load(pred_fn).get_fdata()
            print(gt_fn, pred_fn, gt.shape, pred.shape, flush=True)

            print("Original:", pred.sum(), flush=True)
            # Postprocessing
            pred_post, N = connected_label(pred)
            print("Postprocessed:", (pred_post>0).shape, N, (pred_post>0).sum(), flush=True)
            volume_dict = {i: (pred_post == i).sum() for i in range(1, N+1)}
            volume_indices_sorted = sorted(
                [i for i in range(1, N+1) if i in volume_dict],
                key=lambda i: -volume_dict[i] # large to small sorting
            )        

            if postprocessing_mode == "none":
                print("No postprocessing:", pred.shape, pred.sum(), flush=True)
            elif postprocessing_mode == "percentage":
                pred = np.zeros_like(pred)
                total_volume = sum(volume_dict.values())
                volume_sum = 0
                cur_n_picked_components = 0
                for i in volume_indices_sorted:
                    volume_sum += volume_dict[i]
                    cur_n_picked_components += 1
                    pred += (pred_post == i).astype(int)
                    if volume_sum / total_volume >= n_pick_components: # keep 95% of volume
                        break
                print(volume_indices_sorted[:cur_n_picked_components], flush=True)
                print("After keeping largest {} components:".format(cur_n_picked_components), pred.shape, pred.sum(), flush=True)
            elif postprocessing_mode == "number":
                pred = np.zeros_like(pred)
                print(volume_indices_sorted[:n_pick_components], flush=True)
                for i in range(n_pick_components):
                    pred += (pred_post == volume_indices_sorted[i]).astype(int)
                print("After keeping largest {} components:".format(n_pick_components), pred.shape, pred.sum(), flush=True)
            # Save
            if postprocessing_mode != "none":
                save_path = os.path.join(save_dir, fn + f"_keep-{n_pick_components}.nii.gz")
                if not os.path.exists(save_path):
                    nib.save(nib.Nifti1Image(pred.astype(np.uint8), np.eye(4)), save_path)


            dice = compute_dice(gt, pred)
            precision = compute_precision(gt, pred)
            recall = compute_recall(gt, pred)
            iou = compute_iou(gt, pred)
            hd95 = compute_hausdorff_distance(gt[None,None, ...], pred[None,None, ...], percentile=95).mean().item()
            asd = compute_hausdorff_distance(gt[None,None, ...], pred[None,None, ...], percentile=100).mean().item()
            print(dice, precision, recall, iou, hd95, asd, flush=True)
            write_row = [fn, class_name, dice, precision, recall, iou, hd95, asd]

            # Centerline
            # outputs_sk = skeletonize_3d(pred)
            # labels_sk = skeletonize_3d(gt)
            outputs_sk = skeletonize_to_voxels(pred)
            labels_sk = skeletonize_to_voxels(gt)
            print("skeleton", labels_sk.shape, labels_sk.sum(), flush=True)

            def calc_dice_sub(a, b, smooth=1e-6):
                return ( (a*b).sum() + smooth) / (a.sum() + smooth)

            t_prec = calc_dice_sub(outputs_sk, gt)
            t_sens = calc_dice_sub(labels_sk, pred)
            clDice = (2 * t_prec * t_sens) / (t_prec + t_sens)
            write_row += [t_prec, t_sens, clDice]
            
            writer.writerow(write_row)
            file.flush()

            # debug: 3D visualization

            if DEBUG:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                # subsampling of seg
                def seg_to_points(seg):
                    points = np.argwhere(seg>0)
                    return points
                def subsampling_points(points, n_points=10000):
                    if len(points) > n_points:
                        return points[np.random.choice(len(points), n_points, replace=False)]
                    else:
                        return points
                gt_points = subsampling_points(seg_to_points(gt))
                pred_points = subsampling_points(seg_to_points(pred))
                outputs_sk_points = subsampling_points(seg_to_points(outputs_sk))
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(x=gt_points[:,0], y=gt_points[:,1], z=gt_points[:,2], mode='markers', marker=dict(size=1, color='blue'), name='GT'))
                fig.add_trace(go.Scatter3d(x=pred_points[:,0], y=pred_points[:,1], z=pred_points[:,2], mode='markers', marker=dict(size=1, color='red'), name='Pred'))
                fig.add_trace(go.Scatter3d(x=outputs_sk_points[:,0], y=outputs_sk_points[:,1], z=outputs_sk_points[:,2], mode='markers', marker=dict(size=1, color='green'), name='Pred Skeleton'))
                fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'), title=f"Visualization of {fn} - {class_name}")
                fig.write_html(os.path.join(test_dir, f"visualization_{fn}_{class_name}.html"))
                print(f"Saved visualization for {fn} - {class_name} at {os.path.join(test_dir, f'visualization_{fn}_{class_name}.html')}", flush=True)
                break