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
DEFAULT_PREDICTION_ROOT = Path("/data/b09401064/nnUNet_dataset/nnUNet_results")


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
        return
    destination.symlink_to(source)
    print(f"link {source} -> {destination}", flush=True)


def load_sample_lists(dataset_path: Path) -> tuple[list[str], list[str]]:
    data_list = sorted(child.name for child in dataset_path.iterdir() if child.is_dir())
    train_list = data_list[0::5] + data_list[1::5] + data_list[2::5] + data_list[3::5]
    validation_list = data_list[4::5]
    print(f"All samples: {len(data_list)}", flush=True)
    print(f"Training samples: {len(train_list)}", flush=True)
    print(f"Validation samples: {len(validation_list)}", flush=True)
    return train_list, validation_list


def write_dataset_json(
    raw_root: Path,
    dataset_name: str,
    channel_names: dict[str, str],
    source_dataset_name: str | None = None,
) -> None:
    dataset_root = raw_root / dataset_name
    source_name = source_dataset_name or dataset_name
    source_json_path = raw_root / source_name / "dataset.json"
    if source_json_path.exists():
        with source_json_path.open() as source_file:
            dataset_json = json.load(source_file)
    else:
        dataset_json = {"file_ending": ".nii.gz", "overwrite_image_reader_writer": "NibabelIO"}

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
    validation_samples: Iterable[str],
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
        (list(validation_samples), "imagesTs", "labelsTs"),
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
            if image_file_path.exists() and label_file_path.exists():
                continue

            label = torch.permute(label, (0, 3, 2, 1))
            nib.save(nib.Nifti1Image(image[0].cpu().numpy(), affine), image_file_path)
            nib.save(nib.Nifti1Image(label[0].cpu().numpy(), affine), label_file_path)
            print(f"created {image_file_path} and {label_file_path}", flush=True)


def create_linked_dataset(
    raw_root: Path,
    prediction_root: Path,
    dataset_name: str,
    train_samples: Iterable[str],
    validation_samples: Iterable[str],
    image_sources: tuple[str, str],
    label_source: str,
    channel_names: dict[str, str],
) -> None:
    dataset_root = prepare_directories(raw_root, dataset_name)
    for sample_group, image_directory, label_directory in (
        (train_samples, "imagesTr", "labelsTr"),
        (validation_samples, "imagesTs", "labelsTs"),
    ):
        for sample in sample_group:
            image_sources_for_sample = (
                raw_root / image_sources[0] / image_directory / f"{sample}_0000.nii.gz",
                prediction_root / "Dataset001_Pelvic" / f"{sample}.nii.gz"
                if image_sources[1] == "prediction"
                else raw_root / image_sources[0] / label_directory / f"{sample}.nii.gz"
                if image_sources[1] == "gt"
                else raw_root / image_sources[1] / image_directory / f"{sample}.nii.gz",
            )
            for channel, source in enumerate(image_sources_for_sample):
                link_if_missing(source, dataset_root / image_directory / f"{sample}_000{channel}.nii.gz")
            link_if_missing(
                raw_root / label_source / label_directory / f"{sample}.nii.gz",
                dataset_root / label_directory / f"{sample}.nii.gz",
            )
    write_dataset_json(raw_root, dataset_name, channel_names, label_source)


def create_dataset_006(
    raw_root: Path,
    prediction_root: Path,
    train_samples: Iterable[str],
    validation_samples: Iterable[str],
) -> None:
    dataset_root = prepare_directories(raw_root, "Dataset006_PelvicIPAMixedInput")
    for sample_group, image_directory, label_directory in (
        (train_samples, "imagesTr", "labelsTr"),
        (validation_samples, "imagesTs", "labelsTs"),
    ):
        for variant, vessel_source in (("GT", "Dataset001_Pelvic"), ("Pred", "prediction")):
            for sample in sample_group:
                basename = f"{sample}_{variant}"
                link_if_missing(
                    raw_root / "Dataset001_Pelvic" / image_directory / f"{sample}_0000.nii.gz",
                    dataset_root / image_directory / f"{basename}_0000.nii.gz",
                )
                vessel_path = (
                    prediction_root / "Dataset001_Pelvic" / f"{sample}.nii.gz"
                    if vessel_source == "prediction"
                    else raw_root / vessel_source / image_directory / f"{sample}.nii.gz"
                )
                link_if_missing(
                    vessel_path,
                    dataset_root / image_directory / f"{basename}_0001.nii.gz",
                )
                link_if_missing(
                    raw_root / "Dataset007_IPA" / label_directory / f"{sample}.nii.gz",
                    dataset_root / label_directory / f"{basename}.nii.gz",
                )
    write_dataset_json(
        raw_root,
        "Dataset006_PelvicIPAMixedInput",
        {"0": "CT", "1": "vessels"},
        "Dataset007_IPA",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_directory(args.dataset_path)
    train_samples, validation_samples = load_sample_lists(args.dataset_path)

    convert_dataset(
        args.dataset_path,
        args.raw_root,
        "Dataset001_Pelvic",
        ["vessels"],
        train_samples,
        validation_samples,
    )
    convert_dataset(
        args.dataset_path,
        args.raw_root,
        "Dataset007_IPA",
        ["IPA"],
        train_samples,
        validation_samples,
    )

    create_linked_dataset(
        args.raw_root,
        args.prediction_root,
        "Dataset003_PelvicIPA",
        train_samples,
        validation_samples,
        ("Dataset001_Pelvic", "gt"),
        "Dataset007_IPA",
        {"0": "CT", "1": "vessels"},
    )
    create_linked_dataset(
        args.raw_root,
        args.prediction_root,
        "Dataset005_PelvicIPAPredInput",
        train_samples,
        validation_samples,
        ("Dataset001_Pelvic", "prediction"),
        "Dataset007_IPA",
        {"0": "CT", "1": "vessels"},
    )
    create_dataset_006(args.raw_root, args.prediction_root, train_samples, validation_samples)
    print("Dataset creation completed.", flush=True)


if __name__ == "__main__":
    main()
