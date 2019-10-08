import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from queue import Queue
from utils import param_count

def cuda_available():
	return torch.cuda.is_available()

def list_sum(x):
	if len(x) == 1:
		return x[0]
	elif len(x) == 2:
		return x[0] + x[1]
	else:
		return x[0] + list_sum(x[1:])

def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)
	
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
	"""
	
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class TransitionBlock(nn.Module):
	def __init__(self, layers):
		super(TransitionBlock, self).__init__()
		
		self.layers = nn.ModuleList(layers)
	
	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x
	
	def get_config(self):
		return {
			'name': TransitionBlock.__name__,
			'layers': [
				layer.get_config() for layer in self.layers
			]
		}
	
	@staticmethod
	def set_from_config(config):
		layers = []
		for layer_config in config.get('layers'):
			layer = set_layer_from_config(layer_config)
			layers.append(layer)
		block = TransitionBlock(layers)
		return block
	
	def virtual_forward(self, x, init=False):
		for layer in self.layers:
			x = layer.virtual_forward(x, init)
		return x
	
	def claim_ready(self, nBatch, noise=None):
		for layer in self.layers:
			layer.claim_ready(nBatch, noise)


class BasicBlockWiseConvNet(nn.Module):
	def __init__(self, blocks, classifier):
		super(BasicBlockWiseConvNet, self).__init__()
		self.blocks = nn.ModuleList(blocks)
		self.classifier = classifier
	
	def forward(self, x):
		for block in self.blocks:
			x = block(x)
		x = x.view(x.size(0), -1)  # flatten
		x = self.classifier(x)
		return x
	
	@property
	def building_block(self):
		raise NotImplementedError
	
	def get_config(self):
		raise NotImplementedError
	
	@staticmethod
	def set_from_config(config):
		raise NotImplementedError
	
	@staticmethod
	def set_standard_net(**kwargs):
		raise NotImplementedError
	
	def init_model(self, model_init, init_div_groups):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if model_init == 'he_fout':
					n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
					if init_div_groups:
						n /= m.groups
					m.weight.data.normal_(0, math.sqrt(2. / n))
				elif model_init == 'he_fin':
					n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
					if init_div_groups:
						n /= m.groups
					m.weight.data.normal_(0, math.sqrt(2. / n))
				else:
					raise NotImplementedError
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					m.bias.data.zero_()
	
	def set_non_ready_layers(self, data_loader, nBatch, noise=None, print_info=True):
		self.eval()
		
		top1 = AverageMeter()
		data_loader.drop_last = True
		batch_num = 0
		batch_size = 0
		while batch_num < nBatch:
			for _input, target in data_loader:
				if torch.cuda.is_available():
					target = target.cuda(non_blocking=True)
					_input = _input.cuda()
				input_var = torch.autograd.Variable(_input, volatile=True)
				
				x = input_var
				for block in self.blocks:
					x = block.virtual_forward(x, init=(batch_num == 0))
				x = x.view(x.size(0), -1)  # flatten
				x = self.classifier(x)
				
				acc1 = accuracy(x.data, target, topk=(1,))
				top1.update(acc1[0][0], _input.size(0))
				
				if batch_size == 0:
					batch_size = x.size()[0]
				else:
					assert batch_size == x.size()[0]
				batch_num += 1
				if batch_num >= nBatch:
					break
		if print_info: print(top1.avg)
		for block in self.blocks:
			block.claim_ready(nBatch, noise)
	
	def mimic_run_with_linear_regression(self, data_loader, src_model, sample_size=None,
	                                     distill_epochs=0, distill_lr=0.1, print_info=True):
		src_model.eval()
		input_images, model_outputs = [], []
		for _input, _ in data_loader:
			input_images.append(_input)
			if next(src_model.parameters()).is_cuda:
				_input = _input.cuda()
			input_var = torch.autograd.Variable(_input, volatile=True)
			
			final_logit = src_model(input_var)
			model_outputs.append(final_logit.data.cpu())
			
			if sample_size:
				sample_size -= _input.size(0)
				if sample_size <= 0:
					break
		
		criterion = nn.MSELoss()
		if torch.cuda.is_available():
			criterion = criterion.cuda()
		self.train()
		
		if distill_epochs > 0:
			optimizer = torch.optim.SGD(self.parameters(), lr=distill_lr)
			losses = AverageMeter()
			if print_info: print('start distilling')
			for _j in range(distill_epochs):
				lr = distill_lr
				rand_indexes = np.random.permutation(len(input_images))
				for _i, idx in enumerate(rand_indexes):
					# T_total = distill_epochs * len(rand_indexes)
					# T_cur = _j * len(rand_indexes) + _i
					# lr = 0.5 * distill_lr * (1 + math.cos(math.pi * T_cur / T_total))
					for param_group in optimizer.param_groups:
						param_group['lr'] = lr
					
					_input = input_images[idx]
					target = model_outputs[idx]
					if torch.cuda.is_available():
						input_var = torch.autograd.Variable(_input.cuda())
						target_var = torch.autograd.Variable(target.cuda())
					else:
						input_var = torch.autograd.Variable(_input)
						target_var = torch.autograd.Variable(target)
					self_model_out = self.forward(input_var)
					loss = criterion(self_model_out, target_var)
					
					losses.update(loss.data[0], _input.size(0))
					
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				if print_info: print('epoch [%d]: %f\tlr=%f' % (_j, losses.avg, lr))

def get_block_by_name(name):
	if name == TransitionBlock.__name__:
		return TransitionBlock
	elif name == ResidualBlock.__name__:
		return ResidualBlock
	else:
		raise NotImplementedError

def get_layer_by_name(name):
	if name == ConvLayer.__name__:
		return ConvLayer
	elif name == DepthConvLayer.__name__:
		return DepthConvLayer
	elif name == PoolingLayer.__name__:
		return PoolingLayer
	elif name == IdentityLayer.__name__:
		return IdentityLayer
	elif name == LinearLayer.__name__:
		return LinearLayer
	else:
		raise ValueError('unrecognized layer: %s' % name)


def set_layer_from_config(layer_config):
	layer_name = layer_config.pop('name')
	layer = get_layer_by_name(layer_name)
	layer = layer(**layer_config)
	return layer


def apply_noise(weights, noise=None):
	if noise is None:
		return weights
	else:
		assert isinstance(noise, dict)
	
	noise_type = noise.get('type', 'normal')
	if noise_type == 'normal':
		ratio = noise.get('ratio', 1e-3)
		std = torch.std(weights)
		weights_noise = torch.Tensor(weights.size()).normal_(0, std * ratio)
	elif noise_type == 'uniform':
		ratio = noise.get('ratio', 1e-3)
		_min, _max = torch.min(weights), torch.max(weights)
		width = (_max - _min) / 2 * ratio
		weights_noise = torch.Tensor(weights.size()).uniform_(-width, width)
	else:
		raise NotImplementedError
	if weights.is_cuda:
		weights_noise = weights_noise.cuda()
	return weights + weights_noise


class BasicLayer(nn.Module):
	def __init__(self, in_channels, out_channels,
	             use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(BasicLayer, self).__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		
		self.use_bn = use_bn
		self.act_func = act_func
		self.dropout_rate = dropout_rate
		self.ops_order = ops_order
		
		self.layer_ready = layer_ready
		
		""" batch norm, activation, dropout """
		if self.use_bn:
			if self.bn_before_weight:
				self.bn = nn.BatchNorm2d(in_channels)
			else:
				self.bn = nn.BatchNorm2d(out_channels)
		else:
			self.bn = None
		if act_func == 'relu':
			if self.ops_list[0] == 'act':
				self.activation = nn.ReLU(inplace=False)
			else:
				self.activation = nn.ReLU(inplace=True)
		else:
			self.activation = None
		if self.dropout_rate > 0:
			self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
		else:
			self.dropout = None
	
	@property
	def ops_list(self):
		return self.ops_order.split('_')
	
	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)
	
	@property
	def bn_before_act(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'act':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)
	
	def forward(self, x):
		for op in self.ops_list:
			if op == 'weight':
				x = self.weight_call(x)
			elif op == 'bn':
				if self.bn is not None:
					x = self.bn(x)
			elif op == 'act':
				if self.activation is not None:
					x = self.activation(x)
			else:
				raise ValueError('Unrecognized op: %s' % op)
		if self.dropout is not None:
			x = self.dropout(x)
		return x
	
	def weight_call(self, x):
		raise NotImplementedError
	
	def get_same_padding(self, kernel_size):
		if kernel_size == 1:
			padding = 0
		elif kernel_size == 3:
			padding = 1
		elif kernel_size == 5:
			padding = 2
		elif kernel_size == 7:
			padding = 3
		else:
			raise NotImplementedError
		return padding
	
	def get_config(self):
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'use_bn': self.use_bn,
			'act_func': self.act_func,
			'dropout_rate': self.dropout_rate,
			'ops_order': self.ops_order,
		}
	
	def copy_bn(self, copy_layer, noise=None):
		if noise is None: noise = {}
		if self.use_bn:
			copy_layer.bn.weight.data = self.bn.weight.data.clone()
			copy_layer.bn.bias.data = self.bn.bias.data.clone()
			copy_layer.bn.running_mean = self.bn.running_mean.clone()
			copy_layer.bn.running_var = self.bn.running_var.clone()
	
	def copy(self, noise=None):
		raise NotImplementedError
	
	def split(self, split_list, noise=None):
		raise NotImplementedError
	
	@property
	def get_str(self):
		raise NotImplementedError
	
	def virtual_forward(self, x, init=False):
		if not self.layer_ready:
			if self.use_bn:
				if init:
					self.bn.running_mean.zero_()
					self.bn.running_var.zero_()
				if self.bn_before_act:
					x_ = x
				else:
					x_ = self.activation(x)
				batch_mean = x_
				for dim in [0, 2, 3]:
					batch_mean = torch.mean(batch_mean, dim=dim, keepdim=True)
				batch_var = (x_ - batch_mean) * (x_ - batch_mean)
				for dim in [0, 2, 3]:
					batch_var = torch.mean(batch_var, dim=dim, keepdim=True)
				batch_mean = torch.squeeze(batch_mean)
				batch_var = torch.squeeze(batch_var)
				
				self.bn.running_mean += batch_mean.data
				self.bn.running_var += batch_var.data
			return x
		else:
			return self.forward(x)
	
	def claim_ready(self, nBatch):
		if not self.layer_ready:
			if self.use_bn:
				self.bn.running_mean /= nBatch
				self.bn.running_var /= nBatch
				self.bn.bias.data = self.bn.running_mean.clone()
				self.bn.weight.data = torch.sqrt(self.bn.running_var + self.bn.eps)
			self.layer_ready = True


class ConvLayer(BasicLayer):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
	             use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(ConvLayer, self).__init__(in_channels, out_channels,
		                                use_bn, act_func, dropout_rate, ops_order, layer_ready)
		
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		
		padding = self.get_same_padding(self.kernel_size)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=self.stride,
		                      padding=padding, dilation=self.dilation, groups=self.groups, bias=self.bias)
	
	def weight_call(self, x):
		x = self.conv(x)
		return x
	
	def get_config(self):
		config = {
			'name': ConvLayer.__name__,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dilation': self.dilation,
			'groups': self.groups,
			'bias': self.bias,
		}
		config.update(super(ConvLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		conv_copy = set_layer_from_config(self.get_config())
		# copy weights
		conv_copy.conv.weight.data = apply_noise(self.conv.weight.data.clone(), noise.get('wider'))
		if self.bias:
			conv_copy.conv.bias.data = apply_noise(self.conv.bias.data.clone(), noise.get('wider'))
		self.copy_bn(conv_copy, noise.get('bn'))
		return conv_copy
	
	def split(self, split_list, noise=None):
		assert np.sum(split_list) == self.out_channels
		if noise is None: noise = {}
		
		seg_layers = []
		if self.groups == 1:
			for seg_size in split_list:
				seg_config = self.get_config()
				seg_config['out_channels'] = seg_size
				seg_layers.append(set_layer_from_config(seg_config))
	
			_pt = 0
			for _i in range(len(split_list)):
				seg_size = split_list[_i]
				seg_layers[_i].conv.weight.data = self.conv.weight.data.clone()[_pt:_pt + seg_size, :, :, :]
				if self.bias:
					seg_layers[_i].conv.bias.data = self.conv.bias.data.clone()[_pt:_pt + seg_size]
				if self.use_bn:
					if self.bn_before_weight:
						self.copy_bn(seg_layers[_i], noise.get('bn'))
					else:
						seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
						seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
						seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
						seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
				_pt += seg_size
		else:
			assert self.groups % len(split_list) == 0
			assert np.all([split_list[0] == split_list[_i] for _i in range(1, len(split_list))])
			
			new_groups = self.groups // len(split_list)
			for seg_size in split_list:
				seg_config = self.get_config()
				seg_config['out_channels'] = seg_size
				seg_config['in_channels'] = self.in_channels // len(split_list)
				seg_config['groups'] = new_groups
				seg_layers.append(set_layer_from_config(seg_config))
			
			in_pt, out_pt = 0, 0
			for _i in range(len(split_list)):
				in_seg_size = self.in_channels // len(split_list)
				out_seg_size = split_list[_i]
				seg_layers[_i].conv.weight.data = self.conv.weight.data.clone()[out_pt:out_pt + out_seg_size, :, :, :]
				if self.bias:
					seg_layers[_i].conv.bias.data = self.conv.bias.data.clone()[out_pt:out_pt + out_seg_size]
				if self.use_bn:
					if self.bn_before_weight:
						seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[in_pt:in_pt + in_seg_size]
						seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[in_pt:in_pt + in_seg_size]
						seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[in_pt:in_pt + in_seg_size]
						seg_layers[_i].bn.running_var = self.bn.running_var.clone()[in_pt:in_pt + in_seg_size]
					else:
						seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[out_pt:out_pt + out_seg_size]
						seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[out_pt:out_pt + out_seg_size]
						seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[out_pt:out_pt + out_seg_size]
						seg_layers[_i].bn.running_var = self.bn.running_var.clone()[out_pt:out_pt + out_seg_size]
				out_pt += out_seg_size
				in_pt += in_seg_size
		return seg_layers
	
	@property
	def get_str(self):
		if self.groups == 1:
			return '%dx%d_Conv' % (self.kernel_size, self.kernel_size)
		else:
			return '%dx%d_GroupConv' % (self.kernel_size, self.kernel_size)
	
	def virtual_forward(self, x, init=False):
		if not self.layer_ready and self.bias:
			assert self.ops_order == 'bn_act_weight'
			if init:
				self.conv.bias.data.zero_()
			min_val = x
			for dim in [0, 2, 3]:
				min_val, _ = torch.min(min_val, dim=dim, keepdim=True)
			min_val = torch.squeeze(min_val)
			self.conv.bias.data = torch.min(self.conv.bias.data, min_val.data)
		return super(ConvLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		if noise is None: noise = {}
		if not self.layer_ready:
			super(ConvLayer, self).claim_ready(nBatch)
			if self.bias:
				self.bn.bias.data -= self.conv.bias.data
			
			mid = self.kernel_size // 2
			self.conv.weight.data.zero_()
			weight_init = torch.cat([
				torch.eye(self.conv.weight.size(1)) for _ in range(self.conv.groups)
			], dim=0)
			self.conv.weight.data[:, :, mid, mid] = apply_noise(weight_init, noise.get('deeper'))
		
		assert self.layer_ready


class DepthConvLayer(BasicLayer):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1, bias=False,
	             use_bn=True, act_func='relu', dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(DepthConvLayer, self).__init__(in_channels, out_channels,
		                                     use_bn, act_func, dropout_rate, ops_order, layer_ready)
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation
		self.groups = groups
		self.bias = bias
		
		padding = self.get_same_padding(self.kernel_size)
		self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, stride=self.stride,
		                            padding=padding, dilation=self.dilation, groups=in_channels, bias=False)
		self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=self.groups, bias=self.bias)
	
	def weight_call(self, x):
		x = self.depth_conv(x)
		x = self.point_conv(x)
		return x
	
	def get_config(self):
		config = {
			'name': DepthConvLayer.__name__,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dilation': self.dilation,
			'groups': self.groups,
			'bias': self.bias,
		}
		config.update(super(DepthConvLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		depth_conv_copy = set_layer_from_config(self.get_config())
		# copy weights
		depth_conv_copy.depth_conv.weight.data = apply_noise(self.depth_conv.weight.data.clone(), noise.get('wider'))
		depth_conv_copy.point_conv.weight.data = apply_noise(self.point_conv.weight.data.clone(), noise.get('wider'))
		if self.bias:
			depth_conv_copy.point_conv.bias.data = apply_noise(self.point_conv.bias.data.clone(), noise.get('wider'))
		self.copy_bn(depth_conv_copy, noise.get('bn'))
		return depth_conv_copy
	
	def split(self, split_list, noise=None):
		if noise is None: noise = {}
		assert np.sum(split_list) == self.out_channels
		
		seg_layers = []
		for seg_size in split_list:
			seg_config = self.get_config()
			seg_config['out_channels'] = seg_size
			seg_layers.append(set_layer_from_config(seg_config))
		
		_pt = 0
		for _i in range(len(split_list)):
			seg_size = split_list[_i]
			seg_layers[_i].depth_conv.weight.data = apply_noise(self.depth_conv.weight.data.clone(), noise.get('wider'))
			seg_layers[_i].point_conv.weight.data = self.point_conv.weight.data.clone()[_pt:_pt + seg_size, :, :, :]
			if self.bias:
				seg_layers[_i].point_conv.bias.data = self.point_conv.bias.data.clone()[_pt:_pt + seg_size]
			if self.use_bn:
				if self.bn_before_weight:
					self.copy_bn(seg_layers[_i], noise.get('bn'))
				else:
					seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
					seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
					seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
					seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
			_pt += seg_size
		return seg_layers
	
	@property
	def get_str(self):
		return '%dx%d_DepthConv' % (self.kernel_size, self.kernel_size)
	
	def virtual_forward(self, x, init=False):
		if not self.layer_ready and self.bias:
			assert self.ops_order == 'bn_act_weight'
			if init:
				self.point_conv.bias.data.zero_()
			min_val = x
			for dim in [0, 2, 3]:
				min_val, _ = torch.min(min_val, dim=dim, keepdim=True)
			min_val = torch.squeeze(min_val)
			self.point_conv.bias.data = torch.min(self.point_conv.bias.data, min_val.data)
		return super(DepthConvLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		if noise is None: noise = {}
		if not self.layer_ready:
			super(DepthConvLayer, self).claim_ready(nBatch)
			if self.bias:
				self.bn.bias.data -= self.point_conv.bias.data
			
			mid = self.kernel_size // 2
			self.depth_conv.weight.data.zero_()
			self.depth_conv.weight.data[:, 0, mid, mid].fill_(1)
			self.depth_conv.weight.data = apply_noise(self.depth_conv.weight.data, noise.get('deeper'))
			
			self.point_conv.weight.data.zero_()
			self.point_conv.weight.data[:, :, 0, 0] = torch.eye(self.point_conv.weight.size(0))
			self.point_conv.weight.data = apply_noise(self.point_conv.weight.data, noise.get('deeper'))
		
		assert self.layer_ready


class PoolingLayer(BasicLayer):
	def __init__(self, in_channels, out_channels, pool_type, kernel_size=2, stride=2,
	             use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(PoolingLayer, self).__init__(in_channels, out_channels,
		                                   use_bn, act_func, dropout_rate, ops_order, layer_ready)
		
		self.pool_type = pool_type
		self.kernel_size = kernel_size
		self.stride = stride
		
		if self.stride == 1:
			padding = self.get_same_padding(self.kernel_size)
		else:
			padding = 0
		
		if self.pool_type == 'avg':
			self.pool = nn.AvgPool2d(self.kernel_size, stride=self.stride, padding=padding, count_include_pad=False)
		elif self.pool_type == 'max':
			self.pool = nn.MaxPool2d(self.kernel_size, stride=self.stride, padding=padding)
		else:
			raise NotImplementedError
	
	def weight_call(self, x):
		x = self.pool(x)
		return x
	
	def get_config(self):
		config = {
			'name': PoolingLayer.__name__,
			'pool_type': self.pool_type,
			'kernel_size': self.kernel_size,
			'stride': self.stride,
		}
		config.update(super(PoolingLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		copy_layer = set_layer_from_config(self.get_config())
		self.copy_bn(copy_layer, noise.get('bn'))
		return copy_layer
	
	def split(self, split_list, noise=None):
		if noise is None: noise = {}
		assert np.sum(split_list) == self.out_channels
		
		seg_layers = []
		for seg_size in split_list:
			seg_config = self.get_config()
			seg_config['in_channels'] = seg_size
			seg_config['out_channels'] = seg_size
			seg_layers.append(set_layer_from_config(seg_config))
		
		_pt = 0
		for _i in range(len(split_list)):
			seg_size = split_list[_i]
			if self.use_bn:
				seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
			_pt += seg_size
		return seg_layers
	
	@property
	def get_str(self):
		return '%dx%d_%sPool' % (self.kernel_size, self.kernel_size, self.pool_type.upper())
	
	def virtual_forward(self, x, init=False):
		return super(PoolingLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		super(PoolingLayer, self).claim_ready(nBatch)
		assert self.layer_ready


class IdentityLayer(BasicLayer):
	def __init__(self, in_channels, out_channels,
	             use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act', layer_ready=True):
		super(IdentityLayer, self).__init__(in_channels, out_channels,
		                                    use_bn, act_func, dropout_rate, ops_order, layer_ready)
	
	def weight_call(self, x):
		return x
	
	def get_config(self):
		config = {
			'name': IdentityLayer.__name__,
		}
		config.update(super(IdentityLayer, self).get_config())
		return config
	
	def copy(self, noise=None):
		if noise is None: noise = {}
		copy_layer = set_layer_from_config(self.get_config())
		self.copy_bn(copy_layer, noise.get('bn'))
		return copy_layer
	
	def split(self, split_list, noise=None):
		if noise is None: noise = {}
		assert np.sum(split_list) == self.out_channels
		
		seg_layers = []
		for seg_size in split_list:
			seg_config = self.get_config()
			seg_config['in_channels'] = seg_size
			seg_config['out_channels'] = seg_size
			seg_layers.append(set_layer_from_config(seg_config))
		
		_pt = 0
		for _i in range(len(split_list)):
			seg_size = split_list[_i]
			if self.use_bn:
				seg_layers[_i].bn.weight.data = self.bn.weight.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.bias.data = self.bn.bias.data.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_mean = self.bn.running_mean.clone()[_pt:_pt + seg_size]
				seg_layers[_i].bn.running_var = self.bn.running_var.clone()[_pt:_pt + seg_size]
			_pt += seg_size
		return seg_layers
	
	@property
	def get_str(self):
		return 'Identity'
	
	def virtual_forward(self, x, init=False):
		return super(IdentityLayer, self).virtual_forward(x, init)
	
	def claim_ready(self, nBatch, noise=None):
		super(IdentityLayer, self).claim_ready(nBatch)
		assert self.layer_ready


class LinearLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=True):
		super(LinearLayer, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias
		
		self.linear = nn.Linear(self.in_features, self.out_features, self.bias)
	
	def forward(self, x):
		return self.linear(x)
	
	def get_config(self):
		return {
			'name': LinearLayer.__name__,
			'in_features': self.in_features,
			'out_features': self.out_features,
			'bias': self.bias,
		}

class TreeNode(nn.Module):
	
	SET_MERGE_TYPE = 'set_merge_type'
	INSERT_NODE = 'insert_node'
	REPLACE_IDENTITY_EDGE = 'replace_identity_edge'
	
	def __init__(self, child_nodes, edges, in_channels, out_channels,
	             split_type='copy', merge_type='add', use_avg=True, bn_before_add=False,
	             path_drop_rate=0, use_zero_drop=True, drop_only_add=False, cell_drop_rate=0):
		super(TreeNode, self).__init__()
		
		self.edges = nn.ModuleList(edges)
		self.child_nodes = nn.ModuleList(child_nodes)
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.split_type = split_type
		self.merge_type = merge_type
		
		self.use_avg = use_avg
		self.bn_before_add = bn_before_add
		
		self.path_drop_rate = path_drop_rate
		self.use_zero_drop = use_zero_drop
		self.drop_only_add = drop_only_add
		self.cell_drop_rate = cell_drop_rate
		
		assert len(edges) == len(child_nodes)
		
		self.branch_bns = None
		if self.bn_before_add and self.merge_type == 'add':
			branch_bns = []
			for _i in range(self.child_num):
				branch_bns.append(nn.BatchNorm2d(self.out_dim_list[_i]))
			self.branch_bns = nn.ModuleList(branch_bns)
	
	@property
	def child_num(self):
		return len(self.edges)
	
	@property
	def in_dim_list(self):
		if self.split_type == 'copy':
			in_dim_list = [self.in_channels] * self.child_num
		elif self.split_type == 'split':
			in_dim_list = self.get_split_list(self.in_channels, self.child_num)
		else:
			assert self.child_num == 1
			in_dim_list = [self.in_channels]
		return in_dim_list
	
	@property
	def out_dim_list(self):
		if self.merge_type == 'add':
			out_dim_list = [self.out_channels] * self.child_num
		elif self.merge_type == 'concat':
			out_dim_list = self.get_split_list(self.out_channels, self.child_num)
		else:
			assert self.child_num == 1
			out_dim_list = [self.out_channels]
		return out_dim_list
		
	@staticmethod
	def get_split_list(in_dim, child_num):
		in_dim_list = [in_dim // child_num] * child_num
		for _i in range(in_dim % child_num):
			in_dim_list[_i] += 1
		return in_dim_list
	
	@staticmethod
	def path_normal_forward(x, edge=None, child=None, branch_bn=None, use_avg=False):
		if edge is not None:
			x = edge(x)
		edge_x = x
		if child is not None:
			x = child(x)
		if branch_bn is not None:
			x = branch_bn(x)
			x += edge_x
			if use_avg: x /= 2
		return x
	
	def path_drop_forward(self, x, branch_idx):
		edge, child = self.edges[branch_idx], self.child_nodes[branch_idx]
		branch_bn = None if self.branch_bns is None else self.branch_bns[branch_idx]
		if self.drop_only_add and self.merge_type != 'add':
			apply_drop = False
		else:
			apply_drop = True
		if ((hasattr(edge,'in_channels') and edge.in_channels != edge.out_channels) 
			or (hasattr(edge,'chn_in') and edge.chn_in != edge.chn_out) 
			or edge.__dict__.get('stride', 1) > 1) and not self.use_zero_drop:
			apply_drop = False
		if apply_drop and self.path_drop_rate > 0:
			p = random.uniform(0, 1)
			drop_flag = p < self.path_drop_rate
			if self.training:
				# train
				if self.use_zero_drop:
					if drop_flag:
						batch_size = x.size()[0]
						feature_map_size = x.size()[2:4]
						stride = edge.__dict__.get('stride', 1)
						out_channels = self.out_dim_list[branch_idx]
						padding = torch.zeros(batch_size, out_channels,
						                      feature_map_size[0] // stride, feature_map_size[1] // stride)
						if x.is_cuda:
							padding = padding.cuda()
						path_out = torch.autograd.Variable(padding)
					else:
						path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
						path_out = path_out / (1 - self.path_drop_rate)
				else:
					raise NotImplementedError
			else:
				if self.use_zero_drop:
					path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
				else:
					raise NotImplementedError
		else:
			path_out = self.path_normal_forward(x, edge, child, branch_bn, use_avg=self.use_avg)
		return path_out
	
	def forward(self, x, virtual=False, init=False):
		if self.cell_drop_rate > 0:
			if self.training:
				p = random.uniform(0, 1)
				drop_flag = p < self.cell_drop_rate
				if self.use_zero_drop:
					if drop_flag:
						# drop
						batch_size = x.size()[0]
						feature_map_size = x.size()[2:4]
						stride = self.edges[0].__dict__.get('stride', 1)
						padding = torch.zeros(batch_size, self.out_channels,
						                      feature_map_size[0] // stride, feature_map_size[1] // stride)
						if x.is_cuda:
							padding = padding.cuda()
						return torch.autograd.Variable(padding)
					else:
						# not drop
						backup = self.cell_drop_rate
						self.cell_drop_rate = 0
						output = self.forward(x, virtual, init)  # normal forward
						self.cell_drop_rate = backup
						output = output / (1 - self.cell_drop_rate)
						return output
				else:
					raise NotImplementedError
			else:
				if self.use_zero_drop:
					pass  # normal forward
				else:
					raise NotImplementedError
				
		if self.split_type == 'copy':
			child_inputs = [x] * self.child_num
		elif self.split_type == 'split':
			child_inputs, _pt = [], 0
			for seg_size in self.in_dim_list:
				seg_x = x[:, _pt:_pt+seg_size, :, :].contiguous()
				child_inputs.append(seg_x)
				_pt += seg_size
		else:
			child_inputs = [x]
		
		child_outputs = []
		for branch_idx in range(self.child_num):
			if virtual:
				edge, child = self.edges[branch_idx], self.child_nodes[branch_idx]
				branch_bn = None if self.branch_bns is None else self.branch_bns[branch_idx]
				path_out = child_inputs[branch_idx]
				if edge:
					path_out = edge.virtual_forward(path_out, init)
				if child:
					path_out = child.virtual_forward(path_out, init)
				if branch_bn:
					if init:
						branch_bn.running_mean.zero_()
						branch_bn.running_var.zero_()
					x_ = path_out
					batch_mean = x_
					for dim in [0, 2, 3]:
						batch_mean = torch.mean(batch_mean, dim=dim, keepdim=True)
					batch_var = (x_ - batch_mean) * (x_ - batch_mean)
					for dim in [0, 2, 3]:
						batch_var = torch.mean(batch_var, dim=dim, keepdim=True)
					batch_mean = torch.squeeze(batch_mean)
					batch_var = torch.squeeze(batch_var)
					
					branch_bn.running_mean += batch_mean.data
					branch_bn.running_var += batch_var.data
					# path_out = branch_bn(path_out)
			else:
				path_out = self.path_drop_forward(child_inputs[branch_idx], branch_idx)
			child_outputs.append(path_out)

		if self.merge_type == 'concat':
			output = torch.cat(child_outputs, dim=1)
		elif self.merge_type == 'add':
			output = list_sum(child_outputs)
			if self.use_avg:
				output = output / self.child_num
		else:
			assert len(child_outputs) == 1
			output = child_outputs[0]
		return output
	
	def get_config(self):
		child_configs = []
		for child in self.child_nodes:
			if child is None:
				child_configs.append(None)
			else:
				child_configs.append(child.get_config())
		edge_configs = []
		for edge in self.edges:
			if edge is None or not hasattr(edge,'get_config'):
				edge_configs.append(None)
			else:
				edge_configs.append(edge.get_config())
		return {
			'in_channels': self.in_channels,
			'out_channels': self.out_channels,
			'split_type': self.split_type,
			'merge_type': self.merge_type,
			'use_avg': self.use_avg,
			'bn_before_add': self.bn_before_add,
			'path_drop_rate': self.path_drop_rate,
			'use_zero_drop': self.use_zero_drop,
			'drop_only_add': self.drop_only_add,
			'cell_drop_rate': self.cell_drop_rate,
			'edges': edge_configs,
			'child_nodes': child_configs,
		}
	
	@staticmethod
	def set_from_config(config):
		child_nodes = []
		for child_config in config.pop('child_nodes'):
			if child_config is None:
				child_nodes.append(None)
			else:
				child_nodes.append(TreeNode.set_from_config(child_config))
		edges = []
		for edge_config in config.pop('edges'):
			if edge_config is None:
				edges.append(None)
			else:
				edges.append(set_layer_from_config(edge_config))
		return TreeNode(child_nodes=child_nodes, edges=edges, **config)

	def get_node(self, node_path):
		node = self
		for branch in node_path:
			node = node.child_nodes[branch]
		return node
	
	def apply_transformation(self, node_path, op_type, op_param):
		tree_node = self.get_node(node_path)
		
		if op_type == TreeNode.SET_MERGE_TYPE:
			tree_node.set_merge_type(**op_param)
		elif op_type == TreeNode.INSERT_NODE:
			tree_node.insert_node(**op_param)
		elif op_type == TreeNode.REPLACE_IDENTITY_EDGE:
			tree_node.replace_identity_edge(**op_param)
		else:
			raise NotImplementedError
	
	@property
	def get_str(self):
		if self.child_num > 0:
			children_str = []
			for _i, child in enumerate(self.child_nodes):
				child_str = None if child is None else child.get_str
				children_str.append('%s=>%s' % (self.edges[_i].get_str, child_str))
			children_str = '[%s]' % ', '.join(children_str)
		else:
			children_str = None
		return '{%s, %s, %s}' % (self.merge_type, self.split_type, children_str)
	
	def virtual_forward(self, x, init=False):
		return self.forward(x, virtual=True, init=init)
	
	def claim_ready(self, nBatch, noise=None):
		idx = 0
		for edge, child in zip(self.edges, self.child_nodes):
			branch_bn = None if self.branch_bns is None else self.branch_bns[idx]
			if edge:
				edge.claim_ready(nBatch, noise)
			if child:
				child.claim_ready(nBatch, noise)
			if branch_bn:
				branch_bn.running_mean /= nBatch
				branch_bn.running_var /= nBatch
				branch_bn.bias.data = branch_bn.running_mean.clone()
				branch_bn.weight.data = torch.sqrt(branch_bn.running_var + branch_bn.eps)
			idx += 1
	
	# -------------------------------- transformation operations -------------------------------- #
	
	def set_merge_type(self, merge_type, branch_num, noise=None):
		assert self.merge_type is None, 'current merge type is not None'
		assert self.child_num == 1 and self.child_nodes[0] is None, 'not applicable'
		
		edge = self.edges[0]
		self.merge_type = merge_type
		if merge_type == 'concat':
			split_list = self.get_split_list(edge.out_channels, branch_num)
			if isinstance(edge, IdentityLayer) or isinstance(edge, PoolingLayer):
				self.split_type = 'split'
			elif isinstance(edge, ConvLayer) and edge.groups > 1:
				self.split_type = 'split'
			else:
				self.split_type = 'copy'
			seg_edges = edge.split(split_list, noise)
			self.edges = nn.ModuleList(seg_edges)
		elif merge_type == 'add':
			self.split_type = 'copy'
			copy_edges = [edge.copy()] + [edge.copy(noise) for _ in range(branch_num - 1)]
			self.edges = nn.ModuleList(copy_edges)
		self.child_nodes = nn.ModuleList([None for _ in range(branch_num)])
		
	def insert_node(self, branch_idx):
		assert branch_idx < self.child_num, 'index out of range: %d' % branch_idx
		branch_edge = self.edges[branch_idx]
		branch_node = self.child_nodes[branch_idx]
		identity_edge = IdentityLayer(branch_edge.out_channels, branch_edge.out_channels, use_bn=False, act_func=None,
		                              dropout_rate=0, ops_order=branch_edge.ops_order)
		inserted_node = TreeNode(child_nodes=[branch_node], edges=[identity_edge], in_channels=branch_edge.out_channels,
		                         out_channels=branch_edge.out_channels, split_type=None, merge_type=None,
		                         use_avg=self.use_avg, bn_before_add=self.bn_before_add,
		                         path_drop_rate=self.path_drop_rate, use_zero_drop=self.use_zero_drop,
		                         drop_only_add=self.drop_only_add)
		self.child_nodes[branch_idx] = inserted_node
		
	def replace_identity_edge(self, idx, edge_config):
		assert idx < self.child_num, 'index out of range: %d' % idx
		old_edge = self.edges[idx]
		assert isinstance(old_edge, IdentityLayer), 'not applicable'
		
		edge_config['in_channels'] = old_edge.in_channels
		edge_config['out_channels'] = old_edge.out_channels
		edge_config['layer_ready'] = False
		
		if 'groups' in edge_config:
			groups = edge_config['groups']
			in_channels = edge_config['in_channels']
			while in_channels % groups != 0:
				groups -= 1
			edge_config['groups'] = groups
		new_edge = set_layer_from_config(edge_config)
		self.edges[idx] = new_edge



class ResidualBlock(nn.Module):
	def __init__(self, cell, in_bottle, out_bottle, shortcut, final_bn=False):
		super(ResidualBlock, self).__init__()
		
		self.cell = cell
		self.in_bottle = in_bottle
		self.out_bottle = out_bottle
		self.shortcut = shortcut
		
		if final_bn:
			if self.out_bottle is None:
				out_channels = self.cell.out_channels
			else:
				out_channels = self.out_bottle.out_channels
			self.final_bn = nn.BatchNorm2d(out_channels)
		else:
			self.final_bn = None
	
	def forward(self, x):
		_x = self.shortcut(x)
		
		if self.in_bottle is not None:
			x = self.in_bottle(x)
		
		x = self.cell(x)
		
		if self.out_bottle is not None:
			x = self.out_bottle(x)
		if self.final_bn:
			x = self.final_bn(x)
		
		residual_channel = x.size()[1]
		shortcut_channel = _x.size()[1]
		
		batch_size = x.size()[0]
		featuremap = x.size()[2:4]
		if residual_channel != shortcut_channel:
			padding = torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap[0], featuremap[1])
			if x.is_cuda:
				padding = padding.cuda()
			padding = torch.autograd.Variable(padding)
			_x = torch.cat((_x, padding), 1)
		
		return _x + x
	
	def get_config(self):
		return {
			'name': ResidualBlock.__name__,
			'shortcut': self.shortcut.get_config(),
			'in_bottle': None if self.in_bottle is None else self.in_bottle.get_config(),
			'out_bottle': None if self.out_bottle is None else self.out_bottle.get_config(),
			'final_bn': False if self.final_bn is None else True,
			'cell': self.cell.get_config(),
		}
	
	@staticmethod
	def set_from_config(config):
		if config.get('in_bottle'):
			in_bottle = set_layer_from_config(config.get('in_bottle'))
		else:
			in_bottle = None
		
		if config.get('out_bottle'):
			out_bottle = set_layer_from_config(config.get('out_bottle'))
		else:
			out_bottle = None
			
		shortcut = set_layer_from_config(config.get('shortcut'))
		cell = TreeNode.set_from_config(config.get('cell'))
		final_bn = config.get('final_bn', False)
		
		return ResidualBlock(cell, in_bottle, out_bottle, shortcut, final_bn)
	
	def virtual_forward(self, x, init=False):
		_x = self.shortcut.virtual_forward(x, init)
		
		if self.in_bottle is not None:
			x = self.in_bottle.virtual_forward(x, init)
		x = self.cell.virtual_forward(x, init)
		if self.out_bottle is not None:
			x = self.out_bottle.virtual_forward(x, init)
		if self.final_bn:
			x = self.final_bn(x)
		
		residual_channel = x.size()[1]
		shortcut_channel = _x.size()[1]
		
		batch_size = x.size()[0]
		featuremap = x.size()[2:4]
		if residual_channel != shortcut_channel:
			padding = torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap[0], featuremap[1])
			if x.is_cuda:
				padding = padding.cuda()
			padding = torch.autograd.Variable(padding)
			_x = torch.cat((_x, padding), 1)
		
		return _x + x
	
	def claim_ready(self, nBatch, noise=None):
		if self.in_bottle:
			self.in_bottle.claim_ready(nBatch, noise)
		self.cell.claim_ready(nBatch, noise)
		if self.out_bottle:
			self.out_bottle.claim_ready(nBatch, noise)
		self.shortcut.claim_ready(nBatch, noise)


class FixedTreeCell(nn.Module):
	def __init__(self, C_in, C_out, conv1, conv2,
				 edge_cls, edge_kwargs, tree_node_config):
		super().__init__()
		tree_bn = tree_node_config['bn_before_add']
		tree_node_config['bn_before_add'] = False
		subsubtree11 = TreeNode(child_nodes=[None,None],
						edges=[edge_cls(**edge_kwargs),edge_cls(**edge_kwargs)],
						in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)
		subsubtree12 = TreeNode(child_nodes=[None,None],
						edges=[edge_cls(**edge_kwargs),edge_cls(**edge_kwargs)],
						in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)
		subsubtree21 = TreeNode(child_nodes=[None,None], 
						edges=[edge_cls(**edge_kwargs),edge_cls(**edge_kwargs)],
						in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)
		subsubtree22 = TreeNode(child_nodes=[None,None],
						edges=[edge_cls(**edge_kwargs),edge_cls(**edge_kwargs)],
						in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)
		subtree1 = TreeNode(child_nodes=[subsubtree11, subsubtree12],
						edges=[edge_cls(**edge_kwargs),edge_cls(**edge_kwargs)],
						in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)
		subtree2 = TreeNode(child_nodes=[subsubtree21, subsubtree22],
						edges=[edge_cls(**edge_kwargs),edge_cls(**edge_kwargs)],
						in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)

		tree_node_config['bn_before_add'] = True		
		self.root = TreeNode(child_nodes=[subtree1, subtree2],
						edges=[conv1, conv2], in_channels=C_in,
		                out_channels=C_in, split_type='copy', merge_type='add', **tree_node_config)
		# print('FixedTreeCell: chn_in:{} #p:{:.3f}'.format(C_in, param_count(self)))
	
	def forward(self, x):
		return self.root(x)

	def get_config(self):
		return self.root.get_config()

	
class ProxylessNASNet(BasicBlockWiseConvNet):
	def __init__(self, blocks, classifier, ops_order, tree_node_config, groups_3x3):
		super(ProxylessNASNet, self).__init__(blocks, classifier)
		
		self.ops_order = ops_order
		self.tree_node_config = tree_node_config
		self.groups_3x3 = groups_3x3
	
	@property
	def building_block(self):
		for block in self.blocks:
			if isinstance(block, ResidualBlock):
				return block.cell
	
	def get_config(self):
		return {
			'name': ProxylessNASNet.__name__,
			'ops_order': self.ops_order,
			'tree_node_config': self.tree_node_config,
			'groups_3x3': self.groups_3x3,
			'blocks': [
				block.get_config() for block in self.blocks
			],
			'classifier': self.classifier.get_config(),
		}
	
	@staticmethod
	def set_from_config(config):
		blocks = []
		for block_config in config.get('blocks'):
			block = get_block_by_name(block_config.get('name'))
			tree_node_config = copy.deepcopy(config.get('tree_node_config'))
			if block == ResidualBlock:
				block_config['cell'].update(tree_node_config)
				tree_node_config['bn_before_add'] = False
				tree_node_config['cell_drop_rate'] = 0
				to_updates = Queue()
				for child_config in block_config['cell']['child_nodes']:
					to_updates.put(child_config)
				while not to_updates.empty():
					child_config = to_updates.get()
					if child_config is not None:
						child_config.update(tree_node_config)
						for new_config in child_config['child_nodes']:
							to_updates.put(new_config)
			block = block.set_from_config(block_config)
			blocks.append(block)
		
		classifier_config = config.get('classifier')
		classifier = set_layer_from_config(classifier_config)
		
		ops_order = config.get('ops_order')
		groups_3x3 = config.get('groups_3x3', 1)

		return ProxylessNASNet(blocks, classifier, ops_order, config.get('tree_node_config'), groups_3x3)
	
	@staticmethod
	def set_standard_net(data_shape, n_classes, start_planes, alpha, block_per_group, total_groups, downsample_type,
	                     bottleneck=4, ops_order='bn_act_weight', dropout_rate=0,
	                     final_bn=True, no_first_relu=True, use_depth_sep_conv=False, groups_3x3=1,
						 edge_cls=None, edge_kwargs={}, tree_node_config={}):
		image_channel, image_size = data_shape[0:2]
		
		addrate = alpha / (block_per_group * total_groups)  # add pyramid_net
		
		# initial conv
		features_dim = start_planes
		if ops_order == 'weight_bn_act':
			init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=True, act_func='relu',
			                            dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'act_weight_bn':
			init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=True, act_func=None,
			                            dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'bn_act_weight':
			init_conv_layer = ConvLayer(image_channel, features_dim, kernel_size=3, use_bn=False, act_func=None,
			                            dropout_rate=0, ops_order=ops_order)
		else:
			raise NotImplementedError
		if final_bn:
			init_bn_layer = IdentityLayer(features_dim, features_dim, use_bn=True, act_func=None,
			                              dropout_rate=0, ops_order=ops_order)
			transition2blocks = TransitionBlock([init_conv_layer, init_bn_layer])
		else:
			transition2blocks = TransitionBlock([init_conv_layer])
		blocks = [transition2blocks]
		
		planes = start_planes
		for group_idx in range(total_groups):
			for block_idx in range(block_per_group):
				if group_idx > 0 and block_idx == 0:
					stride = 2
					image_size //= 2
				else:
					stride = 1
				# prepare the residual block
				planes += addrate
				if stride == 1:
					shortcut = IdentityLayer(features_dim, features_dim, use_bn=False, act_func=None,
					                         dropout_rate=0, ops_order=ops_order)
				else:
					if downsample_type == 'avg_pool':
						shortcut = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=2, stride=2,
						                        use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
					elif downsample_type == 'max_pool':
						shortcut = PoolingLayer(features_dim, features_dim, 'max', kernel_size=2, stride=2,
						                        use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
					else:
						raise NotImplementedError
				
				out_plane = int(round(planes))
				if out_plane % groups_3x3 != 0:
					out_plane -= out_plane % groups_3x3  # may change to +=
				if no_first_relu:
					in_bottle = ConvLayer(features_dim, out_plane, kernel_size=1, use_bn=True, act_func=None,
					                      dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					in_bottle = ConvLayer(features_dim, out_plane, kernel_size=1, use_bn=True, act_func='relu',
					                      dropout_rate=dropout_rate, ops_order=ops_order)
				
				if use_depth_sep_conv:
					cell_edge1 = DepthConvLayer(out_plane, out_plane, kernel_size=3, stride=stride, use_bn=True,
					                           act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
					cell_edge2 = DepthConvLayer(out_plane, out_plane, kernel_size=3, stride=stride, use_bn=True,
					                           act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				else:
					cell_edge1 = ConvLayer(out_plane, out_plane, kernel_size=3, stride=stride, groups=groups_3x3,
					                      use_bn=True, act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
					cell_edge2 = ConvLayer(out_plane, out_plane, kernel_size=3, stride=stride, groups=groups_3x3,
					                      use_bn=True, act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				
				edge_kwargs['chn_in'] = (out_plane ,)
				cell = FixedTreeCell(out_plane, out_plane, cell_edge1, cell_edge2, edge_cls, edge_kwargs, tree_node_config)
				
				out_bottle = ConvLayer(out_plane, out_plane * bottleneck, kernel_size=1, use_bn=True,
				                       act_func='relu', dropout_rate=dropout_rate, ops_order=ops_order)
				residual_block = ResidualBlock(cell, in_bottle, out_bottle, shortcut, final_bn=final_bn)
				blocks.append(residual_block)
				features_dim = out_plane * bottleneck
		if ops_order == 'weight_bn_act':
			global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
			                               use_bn=False, act_func=None, dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'act_weight_bn':
			global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
			                               use_bn=False, act_func='relu', dropout_rate=0, ops_order=ops_order)
		elif ops_order == 'bn_act_weight':
			global_avg_pool = PoolingLayer(features_dim, features_dim, 'avg', kernel_size=image_size, stride=image_size,
			                               use_bn=True, act_func='relu', dropout_rate=0, ops_order=ops_order)
		else:
			raise NotImplementedError
		transition2classes = TransitionBlock([global_avg_pool])
		blocks.append(transition2classes)
		
		classifier = LinearLayer(features_dim, n_classes, bias=True)
		
		return ProxylessNASNet(blocks, classifier, ops_order, tree_node_config, groups_3x3)

