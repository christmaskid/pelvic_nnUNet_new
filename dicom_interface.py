import os
import nibabel as nib
import pandas as pd
import numpy as np
import random
import pickle
import torch
from monai.transforms import LoadImage

## SUBROUTINES ##
def isPtid(folder):
        if folder[:2]=='AP':
                return True
        if folder[0] in [str(i) for i in range(10)]:
                return True
        return False

def gen_mask(label_paths, image_shape, class_num=1):
        # mask = np.zeros(image_shape)
        mask = torch.zeros([len(label_paths)] + list(image_shape))
        for i, label_path in enumerate(label_paths):
                data = pd.read_csv(label_path, header=None).to_numpy()
                # for axis in range(3):
                #         print(f"Max & min along axis {axis}: {data[:, axis].max()}, {data[:, axis].min()}", flush=True)
                mask[i, data[:, 0], data[:, 1], data[:, 2]] = class_num
                # for row in data:
                #         z, y, x = row[0], row[1], row[2]
                #         mask[z][y][x] = class_num
        return mask


class DICOM_Interface:
        PREPARE_FOLDER = "/data/b09401064/_batches" #'/tmp/b09401064_batches'
        FOLDER_NAME = "/data/b09401064/labels"

        def __init__(self, dataset_path, prepare_transform, prepare_subfunction=True, label_tags=[], seed=None):
                print("Prepare subfunction", prepare_subfunction)
                self.dataset_path = dataset_path
                self.prepare_subfunction = prepare_subfunction
                self.flag = (seed is not None)
                self.process_id = seed
                self.label_tags = label_tags
                self.prepare_transform = prepare_transform

        def prepare_data(self, data_list, save_labelfn=False):
                dataset_path = self.dataset_path
                if self.flag:
                        print(flag, self.process_id)
                else:
                        self.process_id = str(random.randint(1000000, 9999999))

                if self.prepare_subfunction:
                        process_id = self.process_id

                for i in range(len(data_list)):
                        path = data_list[i]
                        print(f"\rPreparing data: {i}/{len(data_list)}: {path.split('/')[-1]}"+" "*10, flush=True)#, end='')

                        subdirs = os.listdir(os.path.join(dataset_path, path))
                        # search for image folder
                        images = [subdir for subdir in subdirs if isPtid(subdir)]
                        if len(images)!=1:
                                raise ValueError(f"More than two or zero folder(s) in {path} for images: {len(images)} folder(s).")
                        short_image_fn = path.split('/')[-1]+"_"+images[0]
                        image_fn = os.path.join(dataset_path, path, images[0])


                        # search for label folder ({tag}.{ptid})
                        label_paths = []
                        for label_tag in self.label_tags:
                                label_dirs = [subdir for subdir in subdirs if label_tag in subdir]
                                if len(label_dirs)!=1:
                                        raise ValueError(f"More than two or zero folder(s) in {path} for label tag {label_tag}: {len(label_dirs)} folder(s).")
                                label_files = [subdir for subdir in os.listdir(os.path.join(dataset_path, path, label_dirs[0]))]
                                if len(label_files)!=1:
                                        raise ValueError(f"More than two or zero file in {path} for segmentation: {len(label_files)} files.")
                                label_paths.append(os.path.join(dataset_path, path, label_dirs[0], label_files[0]))
                                # ptid = label_dirs[0].split('.')[1]

                        # generate 3D mask from coordinates in label_path (.csv file)
                        # label_fn = os.path.join(self.FOLDER_NAME, path+'.nii.gz')
                        label_fn = os.path.join(self.FOLDER_NAME, path+'.pt')

                        if save_labelfn and os.path.exists(label_fn):
                                print(f"{label_fn} already exists.", flush=True)
                                # mask = LoadImage(reader='nibabelreader')(label_fn)
                                data_list[i] = {'image': image_fn, 'label': label_fn, "image_fn": short_image_fn}
                        else:
                                # read image shape
                                n_slices = len(os.listdir(image_fn))
                                image_shape = (n_slices, 512, 512)
                                mask = gen_mask(label_paths, image_shape)
                                if save_labelfn:
                                        if not os.path.exists(label_fn):
                                                if not os.path.exists(self.FOLDER_NAME):
                                                        os.mkdir(self.FOLDER_NAME)
                                                # n = nib.Nifti1Image(mask, np.eye(4))
                                                # nib.save(n, label_fn)
                                                torch.save(mask, label_fn)

                                        data_list[i] = {'image': image_fn, 'label': label_fn, "image_fn": short_image_fn}
                                        print(data_list[i], flush=True)
                                else:
                                        data_list[i] = {'image': image_fn, 'label': mask, "image_fn": short_image_fn}

                        
                        ## Prepare subfunctions ##
                        # If prepared
                        if self.flag:
                                for key in data_list[i].keys():
                                        # new_file_name = os.path.join(self.PREPARE_FOLDER, process_id+'.'+path+'_'+key+'_'+str(i)+'.pickle')
                                        new_file_name = os.path.join(self.PREPARE_FOLDER, process_id+'.'+path+'_'+key+'_'+str(i)+'.pt')
                                        data_list[i][key] = new_file_name
                                print(data_list[i], flush=True)

                        elif self.prepare_subfunction:
                                try:
                                        os.mkdir(self.PREPARE_FOLDER)
                                except: pass
                                new_data = self.prepare_transform(data_list[i])
                                print(new_data)
                                if isinstance(new_data, list):
                                        new_data = new_data[0]
                                for key in new_data.keys():
                                        # new_file_name = os.path.join(self.PREPARE_FOLDER, process_id+'.'+path+'_'+key+'_'+str(i)+'.pickle')
                                        new_file_name = os.path.join(self.PREPARE_FOLDER, process_id+'.'+path+'_'+key+'_'+str(i)+'.pt')
                                        print(key, new_file_name, new_data[key].shape, flush=True)
                                        torch.save(new_data[key], new_file_name)
                                        # with open(new_file_name, 'wb') as handle:
                                                # pickle.dump(new_data[key], handle, protocol=4)
                                        # np.savez_compressed(new_file_name, new_data)
                                        data_list[i][key] = new_file_name
                                        #print(data_list[i], flush=True)

                print("\rComplete data preparation.", flush=True)
                return data_list

        def cleanup(self):
                if self.prepare_subfunction:
                        for file in os.listdir(self.PREPARE_FOLDER):
                                os.remove(os.path.join(self.PREPARE_FOLDER, file))
                        os.rmdir(self.PREPARE_FOLDER)
