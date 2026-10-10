import torch
import pickle
import time
from monai.transforms import LoadImaged
import nibabel as nib
import numpy as np

DEBUG = False#True#

## DECLARE TRANFORMS ##
def MyLoadImaged(keys, reader=None):
	def load(data):
		global DEBUG
		if DEBUG == True:
			print(data, flush=True)#.keys())
		#print(data[keys], flush=True)
		#start = time.time()
		if reader is None:
			new_data = LoadImaged(keys=keys)(data)
		new_data = LoadImaged(keys=keys, reader=reader)(data)
		#end = time.time()
		#print(f"Loading {data[keys]} takes {end-start} sec(s).", flush=True)
		return new_data
	return load

def NonSense(msg):
	def func(data):
		print(msg, flush=True)
		# print(data, flush=True)
		for key in data:
			if not isinstance(data[key], str):
				print(key, data[key].shape, flush=True)
			else:
				print(key, data[key], flush=True)
		return data
	return func

def LoadPickle(data_path):
	if DEBUG:
		print(data_path, flush=True)
	
	with open(data_path, 'rb') as handle:
		data = pickle.load(handle)
	#print(data, flush=True)
	return data

def LoadPickleDict(data_dict):
	for key in data_dict:
		data_dict[key] = LoadPickle(data_dict[key])
	return data_dict

def LoadTorchPtDict(keys):
	def func(data_dict):
		for key in keys:
			data_dict[key] = torch.load(data_dict[key], weights_only=True)
		return data_dict
	return func

def MyFlipd(keys, spatial_axis):
	def func(data_dict):
		for key in keys:
			print(data_dict[key].shape)
			if 0 in spatial_axis:
				data_dict[key] = data_dict[key][::-1, :, :]
			elif 1 in spatial_axis:
				data_dict[key] = data_dict[key][:, ::-1, :]
			elif 2 in spatial_axis:
				data_dict[key] = data_dict[key][:, :, ::-1]
		return data_dict
	return func
