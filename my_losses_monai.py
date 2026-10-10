# Modified by Yu-Tong Cheng 2022/07/04 -> 2024/08/10: adapted to pytorch version

# ref: https://github.com/frankkramer-lab/MIScnn/blob/master/miscnn/neural_network/metrics.py


import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as distance

# Identify shape of tensor and return correct axes
def identify_axis(shape): 
	# [B, C_in, (L,) W, H] in pytorch; [B, L, W, H, C_in] in tensorflow
	# Three dimensional
	if len(shape) == 5 : return [2,3,4] # <- [1,2,3] in tf
	# Two dimensional
	elif len(shape) == 4 : return [2,3] # <- [1,2] in tf
	# Exception - Unknown
	else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')

#-----------------------------------------------------#
#					Tversky loss					 #
#-----------------------------------------------------#
#					 Reference:					  #
#				Sadegh et al. (2017)				 #
#	 Tversky loss function for image segmentation	#
#	  using 3D fully convolutional deep networks	 #
#-----------------------------------------------------#
# alpha=beta=0.5 : dice coefficient				   #
# alpha=beta=1   : jaccard							#
# alpha+beta=1   : produces set of F*-scores		  #
#-----------------------------------------------------#
class tversky_loss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, y_true, y_pred):
		smooth=0.000001
		# Define alpha and beta
		alpha = 0.5
		beta  = 0.5
		# Calculate Tversky for each class
		axis = identify_axis(y_true.shape)
		tp = torch.sum(y_true * y_pred, axis=axis)
		fn = torch.sum(y_true * (1-y_pred), axis=axis)
		fp = torch.sum((1-y_true) * y_pred, axis=axis)
		tversky_class = (tp + smooth)/(tp + alpha*fn + beta*fp + smooth)
		# Sum up classes to one score
		loss = 1 - torch.mean(tversky_class)
		return loss


#-----------------------------------------------------#
#					Surface loss					 #
#-----------------------------------------------------#
#					 Reference:					  #
#			   Kervadec et al. (2019)				#
#   Boundary loss for highly unbalanced segmentation. #
#-----------------------------------------------------#

# ### https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py

# # Simple script which includes functions for calculating surface loss in keras
# ## See the related discussion: https://github.com/LIVIAETS/boundary-loss/issues/14

def calc_dist_map(seg):
	res = np.zeros_like(seg)
	posmask = seg.astype(np.bool_)
	if posmask.any():
		negmask = ~posmask
		res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
	return res


def calc_dist_map_batch(y_true):
	y_true_numpy = y_true.numpy()
	return np.array([calc_dist_map(y)
					 for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
	y_true_dist_map = calc_dist_map_batch(y_true).astype(np.float32)
	multipled = y_pred * y_true_dist_map
	return torch.mean(multipled)


# # Implementation of Scheduler
# ### The following scheduler was proposed by @marcinkaczor
# ### https://github.com/LIVIAETS/boundary-loss/issues/14#issuecomment-547048076

# In the orignial paper, scheduling is used, gradually increasing the proportion of surface loss (alpha)
# in the combined loss function (tversky loss + surface loss) by 1% every epoch.

# class AlphaScheduler(Callback):
#	 def __init__(self, alpha, update_fn):
#		 super().__init__()
#		 self.alpha = alpha
#		 self.update_fn = update_fn
#	 def on_epoch_end(self, epoch, logs=None):
#		 updated_alpha = self.update_fn(K.get_value(self.alpha))
#		 K.set_value(self.alpha, updated_alpha)


# alpha = K.variable(1, dtype='float32')

def tl_sl_wrapper(alpha):
   def tl_sl(y_true, y_pred):
	   return alpha * tversky_loss(y_true, y_pred) + (1 - alpha) * surface_loss_keras(y_true, y_pred)
   return tl_sl

# model.compile(loss=gl_sl_wrapper(alpha))

def update_alpha(value):
  print(f"alpha updated from {value}")
  return np.clip(value - 0.01, 0.01, 1)

# history = model.fit_generator(
#   ...,
#   callbacks=AlphaScheduler(alpha, update_alpha)
# )


#-----------------------------------------------------#
#			 Symmetric Focal Tversky Loss			#
#-----------------------------------------------------#
#					 Reference:					  #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#			   Leonardo Rundo. (2021)				#
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#			 medical image segmentation			  #
#-----------------------------------------------------#
#				Implementation Source:			   #
#	  https://github.com/mlyg/unified-focal-loss	 #
#-----------------------------------------------------#
def symmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
	"""This is the implementation for binary segmentation.
	Parameters
	----------
	delta : float, optional
		controls weight given to false positive and false negatives, by default 0.7
	gamma : float, optional
		focal parameter controls degree of down-weighting of easy examples, by default 0.75
	"""
	def loss_func(y_true, y_pred):
		# Clip values to prevent division by zero error
		epsilon = torch.finfo().eps # PyTorch equilavent of K.epsilon()
		y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

		axis = identify_axis(y_true.shape)
		# Calculate true positives (tp), false negatives (fn) and false positives (fp)
		tp = torch.sum(y_true * y_pred, axis=axis)
		fn = torch.sum(y_true * (1-y_pred), axis=axis)
		fp = torch.sum((1-y_true) * y_pred, axis=axis)
		dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

		#calculate losses separately for each class, enhancing both classes
		back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -gamma)
		fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma)

		# Average class scores
		loss = torch.mean(torch.stack([back_dice,fore_dice],axis=-1))
		return loss
	return loss_func

#-----------------------------------------------------#
#				Symmetric Focal Loss				 #
#-----------------------------------------------------#
#					 Reference:					  #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#			   Leonardo Rundo. (2021)				#
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#			 medical image segmentation			  #
#-----------------------------------------------------#
#				Implementation Source:			   #
#	  https://github.com/mlyg/unified-focal-loss	 #
#-----------------------------------------------------#
def symmetric_focal_loss(delta=0.7, gamma=2.):
	"""
	Parameters
	----------
	delta : float, optional
		controls weight given to false positive and false negatives, by default 0.7
	gamma : float, optional
		Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
	"""
	def loss_func(y_true, y_pred):

		axis = identify_axis(y_true.shape)

		epsilon = torch.finfo().eps
		y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)
		cross_entropy = -y_true * torch.log(y_pred)
		#calculate losses separately for each class
		back_ce = torch.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
		back_ce =  (1 - delta) * back_ce

		fore_ce = torch.pow(1 - y_pred[:,:,:,1], gamma) * cross_entropy[:,:,:,1]
		fore_ce = delta * fore_ce

		loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce],axis=-1),axis=-1))

		return loss
	return loss_func

#-----------------------------------------------------#
#			 Symmetric Unified Focal Loss			#
#-----------------------------------------------------#
#					 Reference:					  #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#			   Leonardo Rundo. (2021)				#
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#			 medical image segmentation			  #
#-----------------------------------------------------#
#				Implementation Source:			   #
#	  https://github.com/mlyg/unified-focal-loss	 #
#-----------------------------------------------------#
def sym_unified_focal_loss(weight=0.5, delta=0.6, gamma=0.5):
	"""The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
	Parameters
	----------
	weight : float, optional
		represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
	delta : float, optional
		controls weight given to each class, by default 0.6
	gamma : float, optional
		focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
	"""
	def loss_func(y_true, y_pred):
		symmetric_ftl = symmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
		symmetric_fl = symmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
		if weight is not None:
			return (weight * symmetric_ftl) + ((1-weight) * symmetric_fl)
		else:
			return symmetric_ftl + symmetric_fl
	return loss_func


#################################
# Asymmetric Focal Tversky loss #
#################################
def asymmetric_focal_tversky_loss(delta=0.7, gamma=0.75):
	"""This is the implementation for binary segmentation.
	Parameters
	----------
	delta : float, optional
		controls weight given to false positive and false negatives, by default 0.7
	gamma : float, optional
		focal parameter controls degree of down-weighting of easy examples, by default 0.75
	"""
	def loss_func(y_true, y_pred):
		# Clip values to prevent division by zero error
		epsilon = torch.finfo().eps
		y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)

		axis = identify_axis(y_true.shape)
		# Calculate true positives (tp), false negatives (fn) and false positives (fp)	 
		tp = torch.sum(y_true * y_pred, axis=axis)
		fn = torch.sum(y_true * (1-y_pred), axis=axis)
		fp = torch.sum((1-y_true) * y_pred, axis=axis)
		dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

		#calculate losses separately for each class, only enhancing foreground class
		back_dice = (1-dice_class[:,0]) 
		fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -gamma) 

		# Average class scores
		loss = torch.mean(torch.stack([back_dice,fore_dice],axis=-1))
		return loss
	return loss_func

################################
#	 Asymmetric Focal loss	#
################################
class asymmetric_focal_loss(nn.Module):
	def __init__(self, delta=0.7, gamma=2.):
		self.delta = delta
		self.gamma = gamma
	"""For Imbalanced datasets
	Parameters
	----------
	delta : float, optional
		controls weight given to false positive and false negatives, by default 0.7
	gamma : float, optional
		Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
	"""
	def loss_func(y_true, y_pred):
		delta = self.delta
		gamma = self.gamma
		axis = identify_axis(y_true.shape)  

		epsilon = torch.finfo().eps
		y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)
		cross_entropy = -y_true * torch.log(y_pred)

		#calculate losses separately for each class, only suppressing background class
		back_ce = torch.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
		back_ce =  (1 - delta) * back_ce

		fore_ce = cross_entropy[:,:,:,1]
		fore_ce = delta * fore_ce

		loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce],axis=-1),axis=-1))

		return loss

#-----------------------------------------------------#
#			 Symmetric Unified Focal Loss			#
#-----------------------------------------------------#
#					 Reference:					  #
# Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, #
#			   Leonardo Rundo. (2021)				#
#   Unified Focal loss: Generalising Dice and cross   #
#   entropy-based losses to handle class imbalanced   #
#			 medical image segmentation			  #
#-----------------------------------------------------#
#				Implementation Source:			   #
#	  https://github.com/mlyg/unified-focal-loss	 #
#-----------------------------------------------------#
class asym_unified_focal_loss(nn.Module):
	def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
		super().__init__()
		self.weight = weight
		self.delta = delta
		self.gamma = gamma
	"""The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
	Parameters
	----------
	weight : float, optional
		represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
	delta : float, optional
		controls weight given to each class, by default 0.6
	gamma : float, optional
		focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
	"""
	def forward(self, y_true, y_pred):
		asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
		asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)
		if weight is not None:
			return (weight * asymmetric_ftl) + ((1-weight) * asymmetric_fl)  
		else:
			return asymmetric_ftl + asymmetric_fl
