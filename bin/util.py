#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import shutil
import os

## 将整数序列one-hot编码为二进制矩阵
def vectorize_sequences(sequences, dimension=10000):
	# Create an all-zero matrix of shape (len(sequences), dimension)
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.  # set specific indices of results[i] to 1s
	return results


## 计算移动平均值
def smooth_curve(points, factor=0.9):
	smoothed_points = [] 
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1] 
			smoothed_points.append(previous * factor + point * (1 - factor)) 
		else:
			smoothed_points.append(point) 
	return smoothed_points


## 分离dogs vs. cats数据集
def get_small_dataset(original_dataset_dir, base_dir):
	# These directory where we will store our smaller dataset
	os.makedirs(os.path.join(base_dir, 'train', 'cats'), exist_ok=True)
	os.makedirs(os.path.join(base_dir, 'train', 'dogs'), exist_ok=True)
	os.makedirs(os.path.join(base_dir, 'validation', 'cats'), exist_ok=True)
	os.makedirs(os.path.join(base_dir, 'validation', 'dogs'), exist_ok=True)
	os.makedirs(os.path.join(base_dir, 'test', 'cats'), exist_ok=True)
	os.makedirs(os.path.join(base_dir, 'test', 'dogs'), exist_ok=True)
	# Copy first 1000 cat images to train_cats_dir
	fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, "train", fname)
		dst = os.path.join(os.path.join(base_dir, 'train', 'cats', fname))
		shutil.copyfile(src, dst)
	# Copy next 500 cat images to validation_cats_dir
	fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, 'train', fname)
		dst = os.path.join(os.path.join(base_dir, 'validation', 'cats', fname))
		shutil.copyfile(src, dst)
	# Copy next 500 cat images to test_cats_dir
	fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, 'train', fname)
		dst = os.path.join(os.path.join(base_dir, 'test', 'cats', fname))
		shutil.copyfile(src, dst)
	# Copy first 1000 dog images to train_cats_dir
	fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, "train", fname)
		dst = os.path.join(os.path.join(base_dir, 'train', 'dogs', fname))
		shutil.copyfile(src, dst)
	# Copy next 500 dog images to validation_cats_dir
	fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, 'train', fname)
		dst = os.path.join(os.path.join(base_dir, 'validation', 'dogs', fname))
		shutil.copyfile(src, dst)
	# Copy next 500 dog images to test_cats_dir
	fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
	for fname in fnames:
		src = os.path.join(original_dataset_dir, 'train', fname)
		dst = os.path.join(os.path.join(base_dir, 'test', 'dogs', fname))
		shutil.copyfile(src, dst)