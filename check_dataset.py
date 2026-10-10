import os
import numpy as np
import pandas as pd
import csv
import pydicom

if __name__ == "__main__":
	## DECLARE DATASET ##
	# dataset_path = "/data/b09401064/nnUNet_dataset/pelvic_dataset"
	dataset_path = "/data/b09401064/pelvic_dataset_v3-2"
	data_list = sorted(os.listdir(dataset_path))
	print('All samples:', data_list, flush=True)

	label_tags = ["vessels", "IPA"]
	output_table = csv.writer(open("dataset_check_"+os.path.basename(dataset_path)+".csv", "w", newline=""))
	output_table.writerow(["Data Name", "Patient ID", "Patient ID (DICOM)", "Date (DICOM)", "Date (de-id)", "Image Path", "Vessels Path", "IPA Path", "n_slices",
		"vessels voxel count", "IPA voxel count",
		"vessels axis 0 max", "vessels axis 0 min", "vessels axis 1 max", "vessels axis 1 min", "vessels axis 2 max", "vessels axis 2 min",
		"IPA axis 0 max", "IPA axis 0 min", "IPA axis 1 max", "IPA axis 1 min", "IPA axis 2 max", "IPA axis 2 min"])

	err_data = []
	for data_name in data_list:
		print(f"=={data_name}==", flush=True)

		data_path = os.path.join(dataset_path, data_name)
		subdirs = os.listdir(data_path)

		img_paths = []
		label_paths = {label_tag: [] for label_tag in label_tags}

		ptid = None
		for subdir in subdirs:
			for label_tag in label_tags:
				if label_tag in subdir:
					label_paths[label_tag].append(os.path.join(data_path, subdir))
					ptid = subdir.split(".")[-1]
					break
			if subdir == "0.625mm CTA":
				img_paths.append(os.path.join(data_path, subdir))
			elif "." not in subdir:
				ptid = subdir
		if ptid is not None:
			tmp_path = os.path.join(data_path, ptid)
			if os.path.exists(tmp_path):
				img_paths.append(tmp_path)
		
		def format_path(paths):
			if len(paths) == 0:
				return "None"
			elif len(paths) == 1:
				return paths[0]
			else:
				return f"Multiple paths exist: "+",".join(paths)
		
		img_format_path = format_path(img_paths)
		n_slices = None
		print(f"Image path: {img_format_path}", flush=True)
		if len(img_paths) == 1:
			img_path = img_paths[0]
			n_slices = len(os.listdir(img_path))
			print("- n_slices:", n_slices, flush=True)
			
			# try to get date from DICOM
			dicom_files = [f for f in os.listdir(img_paths[0]) if f.endswith(".dcm")]
			if len(dicom_files) > 0:
				dicom_path = os.path.join(img_paths[0], dicom_files[0])
				dicom_data = pydicom.dcmread(dicom_path)
				try:
					# date = dicom_data[0x0008, 0x0020].value
					real_date = dicom_data.get("StudyDate", None)
					ptid_dicom = dicom_data.get("PatientID", None)
					print("Patient ID acquired from DICOM:", ptid_dicom, flush=True)
					print("Study Date acquired from DICOM:", real_date, flush=True)
				except:
					pass

		else:
			img_path = img_format_path
			n_slices = "N/A"
			err_data.append((data_name, "Image path", img_format_path))
			real_date = "N/A"
			ptid_dicom = "N/A"
		
		axis_limits = {}
		voxel_counts = {}
		date = None
		for label_tag in label_tags:
			label_format_path = format_path(label_paths[label_tag])
			if len(label_paths[label_tag]) != 1:
				label_paths[label_tag] = label_format_path
				print(f"Label path for {label_tag}: {label_format_path}", flush=True)
				axis_limits[label_tag] = [("N/A", "N/A")] * 3
				voxel_counts[label_tag] = "N/A"
				err_data.append((data_name, f"{label_tag} path", label_format_path))
				continue
		
			label_dir = label_paths[label_tag][0]
			label_file = os.listdir(label_dir)[0]
			label_path = os.path.join(label_dir, label_file)
			assert label_file.split("_")[0] == ptid, f"Patient ID mismatch: {label_file.split('_')[0]} vs {ptid}"
			if date is None and label_file.split("_")[1] != "seg":
				date = label_file.split("_")[1]
			print(f"Label path for {label_tag}: {label_path}", flush=True)
			label_data = pd.read_csv(label_path, header=None).to_numpy()

			voxel_count = label_data.shape[0]
			print("- voxel count:", voxel_count, flush=True)
			axis_limit = []
			axis_maxes = (n_slices, 512, 512)
			for axis in range(3):
				axis_min = label_data[:, axis].min()
				axis_max = label_data[:, axis].max()
				axis_limit.append((axis_min, axis_max))
				print(f"- axis {axis} min: {axis_min}, max: {axis_max}", flush=True)
				if axis_min < 0 or axis_max >= axis_maxes[axis]:
					print(f"**Warning: out of bounds", flush=True)
					err_data.append((data_name, f"{label_tag} OOB", (axis_min, axis_max, axis_maxes[axis])))

			label_paths[label_tag] = label_path
			axis_limits[label_tag] = axis_limit
			voxel_counts[label_tag] = voxel_count
		print(flush=True)

	# output_table.writerow(["Data Name", "Patient ID", "Date (real)", "Date (de-id)", "Image Path", "Vessels Path", "IPA Path", "n_slices",
	# 	"vessels voxel count", "IPA voxel count",
	# 	"vessels axis 0 max", "vessels axis 0 min", "vessels axis 1 max", "vessels axis 1 min", "vessels axis 2 max", "vessels axis 2 min",
	# 	"IPA axis 0 max", "IPA axis 0 min", "IPA axis 1 max", "IPA axis 1 min", "IPA axis 2 max", "IPA axis 2 min"])

		write_row = [data_name, ptid, ptid_dicom, real_date, date, img_path, label_paths["vessels"], label_paths["IPA"], str(n_slices)]
		write_row += [str(voxel_counts.get(label_tag, "N/A")) for label_tag in ["vessels", "IPA"]]
		for label_tag in ["vessels", "IPA"]:
			for axis in range(3):
				axis_min, axis_max = axis_limits[label_tag][axis]
				write_row.append(str(axis_max))
				write_row.append(str(axis_min))
		output_table.writerow(write_row)

	print("Error data:", flush=True)
	output_table.writerow(["Error data:"])
	for err in err_data:
		print(f" - {err[0]}: {err[1]} - {err[2]}", flush=True)
		output_table.writerow(list(err))

