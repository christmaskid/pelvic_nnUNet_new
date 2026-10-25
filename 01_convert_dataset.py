#!/usr/bin/env python3
"""Convert pelvic DICOM data and create the related nnU-Net datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirstd, Flipd, LoadImaged, ToTensord

from dicom_interface import DICOM_Interface


DEFAULT_DATASET_PATH = Path("/data/b09401064/nnUNet_dataset/pelvic_dataset")
DEFAULT_RAW_ROOT = Path("/data/b09401064/nnUNet_dataset/nnUNet_raw")
DEFAULT_PREDICTION_ROOT = Path("/data/b09401064/nnUNet_dataset/nnUNet_predictions")
DEFAULT_TEST_IDS_PATH = "/home/b09401064/pelvic_CTA/nnUNet_code/csv/test.csv"


class DatasetCreationError(RuntimeError):
    """Raised when the source dataset is not shaped as expected."""


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise DatasetCreationError(f"Missing directory: {path}")


def prepare_directories(raw_root: Path, dataset_name: str) -> Path:
    dataset_root = raw_root / dataset_name
    for directory in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    return dataset_root


def link_if_missing(source: Path, destination: Path) -> None:
    """Create a link even when source is a prediction that will arrive later."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and not destination.exists():
        destination.unlink()
    if destination.exists() or destination.is_symlink():
        print("link already exists:", destination, "->", source, flush=True)
        return
    destination.symlink_to(source)
    print(f"link {source} -> {destination}", flush=True)



def load_sample_lists(dataset_path: Path, assigned_test_ids: list[str], n_split=5) -> tuple[list[str], list[str]]:
    data_list = sorted(child.name for child in dataset_path.iterdir() if child.is_dir())
    if assigned_test_ids:
        test_list = [str(test_id).zfill(4) for test_id in assigned_test_ids]
        test_set = set(test_list)
        train_list = [sample for sample in data_list if sample not in test_set]
    else:
        test_indices = np.arange(len(data_list)) % n_split == (n_split - 1)
        train_list = [sample for i, sample in enumerate(data_list) if not test_indices[i]]
        test_list = [sample for i, sample in enumerate(data_list) if test_indices[i]]
    print(f"All samples: {len(data_list)}: ", data_list, flush=True)
    print(f"Training samples: {len(train_list)}: ", train_list, flush=True)
    print(f"Testing samples: {len(test_list)}: ", test_list, flush=True)
    return train_list, test_list


def write_dataset_json(
    raw_root: Path,
    dataset_name: str,
    channel_names: dict[str, str],
    source_dataset_name: str | None = None,
) -> None:
    dataset_root = raw_root / dataset_name
    source_name = source_dataset_name or dataset_name
    source_json_path = raw_root / source_name / "dataset.json"
    # if source_json_path.exists():
    #     with source_json_path.open() as source_file:
    #         dataset_json = json.load(source_file)
    # else:
    dataset_json = {
        "file_ending": ".nii.gz", 
        "overwrite_image_reader_writer": "NibabelIO",
        "labels": {
            "background": 0,
            "vessels": 1
        },
    }

    dataset_json["numTraining"] = len(list((dataset_root / "imagesTr").glob("*_0000.nii.gz")))
    dataset_json["channel_names"] = channel_names
    with (dataset_root / "dataset.json").open("w") as destination_file:
        json.dump(dataset_json, destination_file, indent=4)
        destination_file.write("\n")


def convert_dataset(
    dataset_path: Path,
    raw_root: Path,
    dataset_name: str,
    label_tags: list[str],
    samples: Iterable[str],
    testing_samples: Iterable[str],
) -> None:
    prepare_transform = Compose(
        [
            LoadImaged(keys="image"),
            ToTensord(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            Flipd(keys=["label"], spatial_axis=0),
        ]
    )
    interface = DICOM_Interface(
        dataset_path,
        prepare_transform,
        prepare_subfunction=False,
        label_tags=label_tags,
    )
    dataset_root = prepare_directories(raw_root, dataset_name)

    for sample_group, image_directory, label_directory in (
        (list(samples), "imagesTr", "labelsTr"),
        (list(testing_samples), "imagesTs", "labelsTs"),
    ):
        data_list = interface.prepare_data(sample_group.copy())
        data_loader = DataLoader(
            Dataset(data=data_list, transform=prepare_transform),
            batch_size=1,
            shuffle=False,
            num_workers=4,
        )
        for sample, data in zip(sample_group, data_loader):
            image = data["image"][0]
            label = data["label"][0]
            affine = np.asarray(image.affine)
            image_file_path = dataset_root / image_directory / f"{sample}_0000.nii.gz"
            label_file_path = dataset_root / label_directory / f"{sample}.nii.gz"
            # if image_file_path.exists() and label_file_path.exists():
            #     continue

            label = torch.permute(label, (0, 3, 2, 1))
            nib.save(nib.Nifti1Image(image[0].cpu().numpy(), affine), image_file_path)
            nib.save(nib.Nifti1Image(label[0].cpu().numpy(), affine), label_file_path)
            print(f"created {image_file_path} and {label_file_path}", flush=True)
    
    write_dataset_json(raw_root, dataset_name, {"0": "CT"})


def create_linked_dataset(
    raw_root: Path,
    prediction_root: Path,
    dataset_name: str,
    train_samples: Iterable[str],
    testing_samples: Iterable[str],
    image_source: str,
    label_source: str,
    variants: list[str],
    channel_names: dict[str, str],
) -> None:
    dataset_root = prepare_directories(raw_root, dataset_name)
    for sample_group, image_directory, label_directory in (
        (train_samples, "imagesTr", "labelsTr"),
        (testing_samples, "imagesTs", "labelsTs"),
    ):
        for sample in sample_group:
            for variant in variants: # ["GT", "Pred"]:
                basename = f"{sample}_{variant}"
                # input CH0: image
                link_if_missing(
                    raw_root / image_source / image_directory / f"{sample}_0000.nii.gz",
                    dataset_root / image_directory / f"{basename}_0000.nii.gz",
                )
                # input CH1: vessel label (GT or Pred)
                if variant == "GT":
                    vessel_source_path = raw_root / "Dataset001_Pelvic" / label_directory / f"{sample}.nii.gz"
                elif variant == "Pred":
                    vessel_source_path = prediction_root / "global" / f"{sample}.nii.gz"
                link_if_missing(
                    vessel_source_path,
                    dataset_root / image_directory / f"{basename}_0001.nii.gz",
                )
                # Label
                link_if_missing(
                    raw_root / label_source / label_directory / f"{sample}.nii.gz",
                    dataset_root / label_directory / f"{basename}.nii.gz",
                )
    write_dataset_json(raw_root, dataset_name, channel_names, label_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--assigned-test-ids-path", type=str, default=DEFAULT_TEST_IDS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_directory(args.dataset_path)
    
    assigned_test_ids_path = Path(args.assigned_test_ids_path)
    assigned_test_ids = []
    if assigned_test_ids_path.is_file():
        import pandas as pd
        df = pd.read_csv(assigned_test_ids_path)
        assigned_test_ids = df['RENAMED_ID'].tolist()
        print("Assigned test IDs loaded from CSV:", assigned_test_ids, flush=True)

    train_samples, testing_samples = load_sample_lists(args.dataset_path, assigned_test_ids)

    # convert_dataset(
    #     args.dataset_path,
    #     args.raw_root,
    #     "Dataset001_Pelvic",
    #     ["vessels"],
    #     train_samples,
    #     testing_samples,
    # )
    # convert_dataset(
    #     args.dataset_path,
    #     args.raw_root,
    #     "Dataset007_IPA",
    #     ["IPA"],
    #     train_samples,
    #     testing_samples,
    # )

    create_linked_dataset(
        raw_root=args.raw_root,
        prediction_root=args.prediction_root,
        dataset_name="Dataset003_PelvicIPA",
        train_samples=train_samples,
        testing_samples=testing_samples,
        image_source="Dataset001_Pelvic",
        label_source="Dataset007_IPA",
        variants=["GT"],
        channel_names={"0": "CT", "1": "noNorm"},
    )
    # create_linked_dataset(
    #     raw_root=args.raw_root,
    #     prediction_root=args.prediction_root,
    #     dataset_name="Dataset005_PelvicIPAPredInput",
    #     train_samples=train_samples,
    #     testing_samples=testing_samples,
    #     image_source="Dataset001_Pelvic",
    #     label_source="Dataset007_IPA",
    #     variants=["Pred"],
    #     channel_names={"0": "CT", "1": "noNorm"},
    # )
    # create_linked_dataset(
    #     raw_root=args.raw_root,
    #     prediction_root=args.prediction_root,
    #     dataset_name="Dataset006_PelvicIPAMixedInput",
    #     train_samples=train_samples,
    #     testing_samples=testing_samples,
    #     image_source="Dataset001_Pelvic",
    #     label_source="Dataset007_IPA",
    #     variants=["GT", "Pred"],
    #     channel_names={"0": "CT", "1": "noNorm"},
    # )
    print("Dataset creation completed.", flush=True)


if __name__ == "__main__":
    main()
