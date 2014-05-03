# Hinstagram: Visualizing Images as Histograms
# --------------------------------------------
# by Jay Hack (jhack@stanford.edu), Spring 2014
from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt
import Image
import seaborn as sns

def make_gaussian_img (height, width, sample_size=(1000000,)):

	gx = np.random.normal (loc=width/2, scale=width/4, size=sample_size).astype(np.int)
	gy = np.random.normal (loc=height/2, scale=height/4, size=sample_size).astype(np.int)
	def in_range (x, dim):
		return x >= 0 and x <= dim
	intensities = Counter([(y, x) for y, x in zip(gy, gx) if in_range(x, width) and in_range(y, height)])
	gaussian_img = np.zeros ((height, width))
	for i in range(height):
		for j in range(width):
			gaussian_img[i][j] = intensities[(i, j)]
	return gaussian_img


if __name__ == '__main__':

	#=====[ Step 1: load image as grayscale	]=====
	image_name = 'IMG_0519.jpg'
	raw_img = 255 - np.array(Image.open (image_name).convert('L').resize((240, 360))) 
	height, width = raw_img.shape

	#=====[ Step 2: *sample* a gaussian distribution	]=====
	gaussian_img = make_gaussian_img (height, width)

	#=====[ Step 3: convolve image with gaussian, normalize	]=====
	convolved_img = np.multiply (raw_img, gaussian_img).astype (np.float)
	convolved_img = convolved_img / (np.max(np.max(convolved_img)) / 255)
	convolved_img = convolved_img.astype (np.uint8)

	#=====[ Step 4: unpack into points - apparently this is the only way seaborn accepts...	]=====
	X, Y = [], []
	for i in range(height):
		for j in range(width):
			Y += [height - i] * convolved_img[i][j]
			X += [j] * convolved_img[i][j]

	#=====[ Step 5: make and display histogram ]=====
	sns.jointplot (np.array(X), np.array(Y), color='#219f85')
	plt.axis ('off')
	plt.show ()



