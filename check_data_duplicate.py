import pandas as pd
from pydicom import dcmread
import os, sys

def find_dicom_dir(data_dir):
    for subdir in os.listdir(data_dir):
        if "." not in subdir:
            return os.path.join(data_dir, subdir)
    return None

def compare_dicom_data(dir1, dir2):
    n1 = len(os.listdir(dir1))
    n2 = len(os.listdir(dir2))
    print(f"Number of DICOM files in {dir1}: {n1}")
    print(f"Number of DICOM files in {dir2}: {n2}")
    if n1 != n2:
        print(f"Different number of files in {dir1} and {dir2}")
        return False

    img_same = True
    for i, file in enumerate(sorted(os.listdir(dir1))):
        file1 = os.path.join(dir1, file)
        file2 = os.path.join(dir2, file)
        if not os.path.exists(file2):
            print(f"File {file} not found in {dir2}")
            return False

        dicom1 = dcmread(file1)
        dicom2 = dcmread(file2)

        # Compare relevant DICOM attributes
        if i==0: # Only print for the first file
            attributes_to_compare = ["PatientID", "StudyDate"]
            for attr in attributes_to_compare:
                if getattr(dicom1, attr, None) != getattr(dicom2, attr, None):
                    print(f"Different {attr} in {file}: {getattr(dicom1, attr, None)} vs {getattr(dicom2, attr, None)}")

        # Compare pixel data content
        if dicom1.pixel_array.tobytes() != dicom2.pixel_array.tobytes():
            print(f"Different pixel data in {file}")
            # return False
            img_same = False

    # return True
    return img_same

def find_label_dir(data_dir, label_tag):
    for subdir in os.listdir(data_dir):
        if label_tag in subdir:
            return os.path.join(data_dir, subdir)
    return None

def find_label_csv(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if len(csv_files) == 1:
        return os.path.join(data_dir, csv_files[0])
    return None

def compare_label_csv(csv1_path, csv2_path):
    df1 = pd.read_csv(csv1_path, header=None)
    df2 = pd.read_csv(csv2_path, header=None)

    if df1.shape != df2.shape:
        print(f"Different shapes: {df1.shape} vs {df2.shape}")
        return False

    if not df1.equals(df2):
        print(f"Different content in CSV files: {csv1_path} vs {csv2_path}")
        return False

    return True

if __name__ == "__main__":

    dataset_path = "/data/b09401064/pelvic_dataset_v3_IPA"
    # data_name_1 = "CVAI_Train_68"
    # data_name_2 = "CVAI_Train_69"
    data_name_1 = "CVAI_Train_" + sys.argv[1]
    data_name_2 = "CVAI_Train_" + sys.argv[2]

    # Image comparison
    dicom_dir_1 = find_dicom_dir(os.path.join(dataset_path, data_name_1))
    dicom_dir_2 = find_dicom_dir(os.path.join(dataset_path, data_name_2))
    if dicom_dir_1 and dicom_dir_2:
        print(f"Comparing DICOM data between {data_name_1} and {data_name_2}")
        res = compare_dicom_data(dicom_dir_1, dicom_dir_2)
        if res:
            print("DICOM data is identical.")
        else:
            print("DICOM data differs.")

    # Label comparison
    label_tags = ["vessels", "IPA"]
    for label_tag in label_tags:
        label_dir_1 = find_label_dir(os.path.join(dataset_path, data_name_1), label_tag)
        label_dir_2 = find_label_dir(os.path.join(dataset_path, data_name_2), label_tag)
        if label_dir_1 and label_dir_2:
            print(f"Comparing {label_tag} labels between {data_name_1} and {data_name_2}")
            label_csv_1 = find_label_csv(label_dir_1)
            label_csv_2 = find_label_csv(label_dir_2)
            if label_csv_1 and label_csv_2:
                print(f"Comparing {label_tag} labels between {data_name_1} and {data_name_2}")
                res = compare_label_csv(label_csv_1, label_csv_2)
                if res:
                    print(f"{label_tag} labels are identical.")
                else:
                    print(f"{label_tag} labels differ.")
            else:
                print(f"Label CSV file for {label_tag} not found in one or both datasets.")
        else:
            print(f"Label directory for {label_tag} not found in one or both datasets.")


    