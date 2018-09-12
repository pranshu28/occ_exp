import pandas as pd
from scipy import signal
from sklearn.model_selection import train_test_split, StratifiedKFold

from cnn_viz import occ, plot
from model import *
from read_data import get_df
from keras.utils import plot_model


def get_XY(new=False):
	if new:
		n_path = "/home/pranshu/Documents/Projects/IITM/AM/temp_data/wafer/normal"
		ab_path = "/home/pranshu/Documents/Projects/IITM/AM/temp_data/wafer/abnormal"
		normal = get_df(n_path)
		abnormal = get_df(ab_path).drop(['box'], axis=1)

		df = normal.append(abnormal, ignore_index=True).sample(frac=1).reset_index(drop=True)
		print "DF ready", normal.shape, abnormal.shape
		df.to_csv("numpy/df.csv")

		df.loc[df[df.ann != 'normal'].index, 'ann'] = 'abnormal'
		df = df.drop(['run', 'wafer'], axis=1)

		X_temp = df.drop(['ann'], axis=1)
		sensors = X_temp.columns

		# le = preprocessing.LabelEncoder()
		# le.fit(df.ann)
		Y = pd.get_dummies(df.ann)  # le.transform(df.ann) #
		classes = Y.columns  # le.classes_ #

		sample = 140
		X = np.array(X_temp)
		a = np.zeros((X.shape[0], X.shape[1], sample))
		for i in range(len(X)):
			for j in range(len(X[0])):
				row = [int(k) for k in X[i, j].split(",")]
				a[i, j, :] = signal.resample(row, sample)

		np.save("numpy/X.npy", a)
		np.save("numpy/Y.npy", Y)
		np.save("numpy/sensors.npy", sensors)
		np.save("numpy/class.npy", classes)

	return pd.read_csv("numpy/df.csv"), np.load("numpy/X.npy").transpose(0, 2, 1), np.load("numpy/Y.npy"), np.load(
		"numpy/sensors.npy"), np.load("numpy/class.npy")


def ExpModel(path, train=False, test=False):
	M = d[path](X, Y, sensors)
	# M.summary()
	if train:
		M = train_model(M, trainx, trainy, path, False)
	M = from_file(M, path)
	if test:
		test_model(M, testx, testy)
	return M


def first_train(paths):
	for i in paths:
		model_ = ExpModel(i, True, True)
		plot_model(model_, to_file=i.split(".")[0] + '.png')
	plt.show()


def cross_val(paths):
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
	for path in paths:
		model_ = ExpModel(path, False, True)
		y = [np.argmax(i) for i in Y]
		occ_iter(model_,path,y,1)
		cvscores = []
		for train, test in kfold.split(X, y):
			scores = model_.evaluate(X[test], Y[test], verbose=0)
			# print "%s: %.2f%%" % (model_.metrics_names[1], scores[1]*100))
			cvscores.append(scores[1] * 100)
		print "\nCross Validation: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))


def occ_iter(model_, path, y, ex):
	name = path.split("/")[-1].split(".")[0]
	ab, nm = [i for i, e in enumerate(y) if e == 0], [i for i, e in enumerate(y) if e == 1]
	for i in ab[:ex] + nm[:ex]:
		occ_exp(i, model_, name)


def occ_exp(index, model_, name, occ_size=5, occ_pixel=0, occ_stride=2):
	inpx, inpy = X[index:index + 1], Y[index:index + 1]
	row = df.iloc[index]
	time_steps = len(inpx[0])
	plot(row, name, sensors, row.ann)

	pred_y = model_.predict(inpx)

	print "\nIndex:", index
	print "Run,wafer: ", df.iloc[index].run, df.iloc[index].wafer
	print '\nPredicted :', classes[pred_y.argmax()]

	probs = occ(model_, inpx, time_steps, occ_size, occ_pixel, occ_stride, pred_y)
	plot(row, name, sensors, row.ann, classes[pred_y.argmax()], probs)


if __name__ == '__main__':
	df, X, Y, sensors, classes = get_XY(False)

	trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.25, random_state=123)
	d = {"models/conv.hdf5": conv_model, "models/lstm.hdf5": lstm_model, "models/lstm6.hdf5": lstm6_model, "models/hybrid.hdf5": hybrid_model}
	paths = [i for i in d.keys()][0:1]
	print '\n\n', trainx.shape, trainy.shape, testx.shape, testy.shape
	print paths
	# first_train(paths)
	cross_val(paths)
	# plt.show()
