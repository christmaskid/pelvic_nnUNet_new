#!/usr/bin/env python3
"""Create nnU-Net datasets 005, 006, and 007 from existing data and predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_PATH = Path("/data/b09401064/nnUNet_dataset/pelvic_dataset")
DEFAULT_RAW_ROOT = Path("/data/b09401064/nnUNet_dataset/nnUNet_raw")
DEFAULT_PREDICTION_ROOT = Path("/data/b09401064/nnUNet_dataset/nnUNet_results/")


class DatasetCreationError(RuntimeError):
    """Raised when the source dataset is not shaped as expected."""


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise DatasetCreationError(f"Missing directory: {path}")


def link_if_missing(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(source)
    print(f"link {source} -> {destination}", flush=True)


def find_image_name(dataset_path: Path, sample: str) -> str:
    sample_path = dataset_path / sample
    candidates = sorted(
        child.name
        for child in sample_path.iterdir()
        if child.is_dir() and (child.name.startswith("AP") or child.name[0].isdigit())
    )
    if len(candidates) != 1:
        raise DatasetCreationError(
            f"Expected one image folder in {sample_path}, found {len(candidates)}: {candidates}"
        )
    return f"{sample}_{candidates[0]}"


def load_sample_lists(dataset_path: Path) -> tuple[list[str], list[str]]:
    data_list = sorted(child.name for child in dataset_path.iterdir() if child.is_dir())
    train_list = data_list[0::5] + data_list[1::5] + data_list[2::5] + data_list[3::5]
    validation_list = data_list[4::5]
    print(f"All samples: {len(data_list)}", flush=True)
    print(f"Training samples: {len(train_list)}", flush=True)
    print(f"Validation samples: {len(validation_list)}", flush=True)
    return train_list, validation_list


def prepare_directories(raw_root: Path, dataset_name: str) -> Path:
    dataset_root = raw_root / dataset_name
    for directory in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    return dataset_root


def write_dataset_json(
    raw_root: Path,
    dataset_name: str,
    label_dataset_name: str,
    channel_names: dict[str, str],
) -> None:
    dataset_root = raw_root / dataset_name
    source_json = raw_root / label_dataset_name / "dataset.json"
    image_training_files = list((dataset_root / "imagesTr").glob("*_0000.nii.gz"))

    with source_json.open() as source_file:
        dataset_json = json.load(source_file)
    dataset_json["numTraining"] = len(image_training_files)
    dataset_json["channel_names"] = channel_names

    with (dataset_root / "dataset.json").open("w") as destination_file:
        json.dump(dataset_json, destination_file, indent=4)
        destination_file.write("\n")


def create_dataset_005(
    raw_root: Path,
    dataset_path: Path,
    prediction_root: Path,
    train_list: Iterable[str],
    validation_list: Iterable[str],
) -> None:
    dataset_name = "Dataset005_PelvicIPAPredInput"
    source_image_dataset = "Dataset001_Pelvic"
    label_dataset = "Dataset003_PelvicIPA"
    dataset_root = prepare_directories(raw_root, dataset_name)

    for samples, image_directory, label_directory in (
        (train_list, "imagesTr", "labelsTr"),
        (validation_list, "imagesTs", "labelsTs"),
    ):
        for sample in samples:
            image_name = find_image_name(dataset_path, sample)
            link_if_missing(
                raw_root / source_image_dataset / image_directory / f"{image_name}_0000.nii.gz",
                dataset_root / image_directory / f"{image_name}_0000.nii.gz",
            )
            link_if_missing(
                prediction_root / f"{image_name}.nii.gz",
                dataset_root / image_directory / f"{image_name}_0001.nii.gz",
            )
            link_if_missing(
                raw_root / label_dataset / label_directory / f"{image_name}.nii.gz",
                dataset_root / label_directory / f"{image_name}.nii.gz",
            )

    write_dataset_json(raw_root, dataset_name, label_dataset, {"0": "CT", "1": "noNorm"})


def create_dataset_006(
    raw_root: Path,
    dataset_path: Path,
    train_list: Iterable[str],
    validation_list: Iterable[str],
) -> None:
    dataset_name = "Dataset006_PelvicIPAMixedInput"
    source_datasets = ("Dataset003_PelvicIPA", "Dataset005_PelvicIPAPredInput")
    label_dataset = "Dataset003_PelvicIPA"
    dataset_root = prepare_directories(raw_root, dataset_name)

    for samples, image_directory, label_directory in (
        (train_list, "imagesTr", "labelsTr"),
        (validation_list, "imagesTs", "labelsTs"),
    ):
        for source_index, source_dataset in enumerate(source_datasets):
            if image_directory == "imagesTs" and source_index == 1:
                continue
            for sample in samples:
                image_name = find_image_name(dataset_path, sample)
                suffix = f"_{source_index}"
                link_if_missing(
                    raw_root / source_dataset / image_directory / f"{image_name}_0000.nii.gz",
                    dataset_root / image_directory / f"{image_name}{suffix}_0000.nii.gz",
                )
                link_if_missing(
                    raw_root / source_dataset / image_directory / f"{image_name}_0001.nii.gz",
                    dataset_root / image_directory / f"{image_name}{suffix}_0001.nii.gz",
                )
                link_if_missing(
                    raw_root / label_dataset / label_directory / f"{image_name}.nii.gz",
                    dataset_root / label_directory / f"{image_name}{suffix}.nii.gz",
                )

    write_dataset_json(raw_root, dataset_name, label_dataset, {"0": "CT", "1": "noNorm"})


def link_matching_files(source_directory: Path, target_directory: Path, pattern: str) -> None:
    require_directory(source_directory)
    for source_file in sorted(source_directory.glob(pattern)):
        link_if_missing(source_file, target_directory / source_file.name)


def create_dataset_007(raw_root: Path) -> None:
    dataset_name = "Dataset007_PelvicIPA_ablation"
    source_dataset = "Dataset003_PelvicIPA"
    dataset_root = prepare_directories(raw_root, dataset_name)

    for directory in ("imagesTr", "imagesTs"):
        link_matching_files(
            raw_root / source_dataset / directory,
            dataset_root / directory,
            "*_0000.nii.gz",
        )
    for directory in ("labelsTr", "labelsTs"):
        link_matching_files(
            raw_root / source_dataset / directory,
            dataset_root / directory,
            "*.nii.gz",
        )

    write_dataset_json(raw_root, dataset_name, source_dataset, {"0": "CT"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_directory(args.dataset_path)
    require_directory(args.raw_root / "Dataset001_Pelvic")
    require_directory(args.raw_root / "Dataset003_PelvicIPA")

    train_list, validation_list = load_sample_lists(args.dataset_path)
    create_dataset_005(
        args.raw_root,
        args.dataset_path,
        args.prediction_root,
        train_list,
        validation_list,
    )
    create_dataset_006(args.raw_root, args.dataset_path, train_list, validation_list)
    create_dataset_007(args.raw_root)
    print("Dataset creation completed.", flush=True)


if __name__ == "__main__":
    main()
