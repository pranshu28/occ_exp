import math

from scipy import signal

from model import *
from matplotlib import cm


def plot(inp, name, sensors, y, pred=None, probs=None):
	f, axarr = plt.subplots(3, 2, figsize=(12, 8))
	f.subplots_adjust(hspace=0.3)
	file = str(inp.run) + "_" + str(inp.wafer)
	title = file + "\nann: " + y
	if probs is not None:
		file = 'occ_' + name + '_' + file
		title = name + " for " + title + "\npredicted: " + pred
	f.suptitle(title, fontsize=12)
	cmap = cm.get_cmap('viridis')
	for col, ax in enumerate(axarr.flat):
		data = [int(i) for i in inp[sensors[col]].split(",")]
		if probs is not None:
			colors = []
			for value in signal.resample(probs[col], len(data)).clip(0, 1):
				colors.append(cmap(value))
		else:
			colors = 'b'
		sc = ax.scatter(range(1, len(data) + 1), data, marker='.', c=colors)
		# sc = ax.imshow([data], cmap=cmap)
		ax.set_title(sensors[col], fontsize=12)

	# f.subplots_adjust(right=0.8)
	# cbar_ax = f.add_axes([0.85, 0.15, 0.01, 0.7])
	# f.colorbar(sc, cax=cbar_ax)
	f.savefig('occ_exp/' + file + '.png')
	return f


def occ(model, inpx, time_steps, occ_size, occ_pixel, occ_stride, classes):
	class_index = np.argmax(classes)
	output_width = int(math.ceil((time_steps - occ_size) / occ_stride + 1))
	output_height = inpx.shape[2]
	print '\nTotal iterations:', output_height, '*', output_width, '=', output_height * output_width

	prob_matrix = np.zeros((output_height, output_width))

	for h in range(output_height):
		for w in range(output_width):
			w_start = w * occ_stride
			w_end = min(time_steps, w_start + occ_size)
			occ_image = inpx.copy()
			occ_image[0, w_start:w_end, h] = occ_pixel
			predictions = model.predict(occ_image)
			prob_matrix[h, w] = predictions[0, class_index]
		# print 'Percentage done :', round(((h + 1) * output_width) * 100 / (output_height * output_width), 2), '%'

	return prob_matrix
