#!/usr/bin/env bash
set -euo pipefail

base=/data/b09401064/nnUNet_dataset/pelvic_dataset
source_dirs=(
    /data/b09401064/pelvic_dataset_v3
    /data/b09401064/pelvic_dataset_v3-2
)

remove_list=(
    CVAI_Train_01
    CVAI_Train_09
    CVAI_Train_19
    CVAI_Train_20
    CVAI_Train_27
    CVAI_Train_34
    CVAI_Train_36
    CVAI_Train_37
    CVAI_Train_42
    CVAI_Train_50
    CVAI_Train_99
    CVAI_Train_0102
    CVAI_Train_0106
)

mkdir -p "$base"

# Remove only links from the target directory; source data is never modified.
find "$base" -maxdepth 1 -mindepth 1 -type l -delete

# Link each source entry explicitly so names from both source directories are preserved.
for source_dir in "${source_dirs[@]}"; do
    while IFS= read -r source_path; do
        source_name=$(basename "$source_path")
        target_path="$base/$source_name"
        if [[ -e "$target_path" || -L "$target_path" ]]; then
            printf 'Error: duplicate target name: %s\n' "$target_path" >&2
            exit 1
        fi
        ln -s "$source_path" "$target_path"
    done < <(find "$source_dir" -maxdepth 1 -mindepth 1 -name 'CVAI_Train_*' -print | sort -V)
done
for name in "${remove_list[@]}"; do
    rm "$base/$name"
done

printf 'Before renaming: %s links\n' "$(find "$base" -maxdepth 1 -mindepth 1 -type l -name 'CVAI_Train_*' | wc -l)"

# First rename all CVAI_Train_* symlinks to CVAI_Train_0xxx for sorting
for src in "$base"/CVAI_Train_*; do
    if [ -L "$src" ]; then
        src_name=$(basename "$src")
        source_number=${src_name#CVAI_Train_}
        printf -v dst_name 'CVAI_Train_%04d' "$((10#$source_number))"
        dst="$base/$dst_name"
        if [[ "$src" != "$dst" ]]; then
            [[ ! -e "$dst" && ! -L "$dst" ]]
            printf '%s -> %s\n' "$src_name" "$dst_name"
            mv -- "$src" "$dst"
        fi
    fi
done

remaining_short=$(find "$base" -maxdepth 1 -mindepth 1 -type l \
    -name 'CVAI_Train_*' ! -name 'CVAI_Train_[0-9][0-9][0-9][0-9]' -print -quit)
if [[ -n "$remaining_short" ]]; then
    printf 'Error: link was not normalized: %s\n' "$remaining_short" >&2
    exit 1
fi
echo "All CVAI_Train_* links normalized to 4-digit numbers."
# ls -l "$base"/CVAI_Train_*

# Sort and rename all symlinks to 4-digit numbers
counter=1
while IFS= read -r src_name; do
    src="$base/$src_name"
    printf -v dst_name '%04d' "$counter"
    dst="$base/$dst_name"
    [ -L "$src" ]
    [ ! -e "$dst" ] && [ ! -L "$dst" ]
    mv -- "$src" "$dst"
    counter=$((counter + 1))
done < <(find "$base" -maxdepth 1 -mindepth 1 -type l -name 'CVAI_Train_[0-9]*' -printf '%f\n' | sort -V)
printf 'Final link count: %d\n' "$((counter - 1))"

printf 'numeric symlinks: '; find "$base" -maxdepth 1 -mindepth 1 -type l -name '[0-9][0-9][0-9][0-9]' -printf '%f\n' | wc -l; \
printf 'CVAI symlinks: '; find "$base" -maxdepth 1 -mindepth 1 -type l -name 'CVAI_Train_*' -printf '%f\n' | wc -l

find "$base" -maxdepth 1 -mindepth 1 -type l -printf '%f\t%l\n' | sort -V