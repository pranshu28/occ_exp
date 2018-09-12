import matplotlib.pylab as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, TimeDistributed, LSTM
from keras.layers import Input, Dense, Reshape, Lambda, Add
from keras.layers import MaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report


def hybrid_model(x, y, sensors):
	out_dim = 1 if len(y.shape) == 1 else y.shape[1]
	inp = Input(shape=(x.shape[1], x.shape[2]), dtype='float')
	reinp = Reshape((x.shape[1], x.shape[2], 1))(inp)
	CNN_out = TimeDistributed(Conv1D(8, 3, activation='relu'))(reinp)
	CNN_out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(CNN_out)
	CNN_out = TimeDistributed(MaxPooling1D(pool_size=3, strides=None, padding='valid'))(CNN_out)
	CNN_out = Reshape((int(CNN_out.shape[1]), int(CNN_out.shape[3])))(CNN_out)
	# LSTM_out = LSTM(32, return_sequences=True)(CNN_out)
	# LSTM_out = LSTM(32,return_sequences=True)(LSTM_out)
	LSTM_out = LSTM(32)(CNN_out)
	out = Dense(out_dim, activation='softmax')(LSTM_out)
	model = Model(inp, out)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	print "\n\n\nInput --> Time(Conv1d(8_3)) --> LSTM(32) --> Dense --> Output"
	return model


def conv_model(x, y, sensors):
	out_dim = 1 if len(y.shape) == 1 else y.shape[1]

	inp = Input(shape=(x.shape[1], x.shape[2]), dtype='float')
	reinp = Reshape((x.shape[1], x.shape[2]))(inp)

	CNN_out = Conv1D(8, 3, activation='relu')(reinp)
	CNN_out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(CNN_out)
	CNN_out = MaxPooling1D(pool_size=3, strides=None, padding='valid')(CNN_out)
	CNN_out = Conv1D(16, 3, activation='relu')(CNN_out)
	CNN_out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(CNN_out)
	CNN_out = MaxPooling1D(pool_size=3, strides=None, padding='valid')(CNN_out)
	CNN_out = Conv1D(16, 3, activation='relu')(CNN_out)
	CNN_out = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(CNN_out)
	CNN_out = GlobalAveragePooling1D()(CNN_out)
	out = Dense(out_dim, activation='softmax')(CNN_out)
	model = Model(inp, out)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	print "\n\n\nInput --> Conv1d(8_3,16_3,16_3) --> Dense --> Output"
	return model


def lstm6_model(x, y, sensors):
	out_dim = 1 if len(y.shape) == 1 else y.shape[1]
	inp = Input(shape=(x.shape[1], x.shape[2]), dtype='float')
	LSTMs = []
	for i, s in enumerate(sensors):
		LSTM_out = Lambda(lambda l: l[:, :, i], output_shape=(x.shape[1],))(inp)
		LSTM_out = Reshape((x.shape[1], 1))(LSTM_out)
		# LSTM_out = LSTM(8, return_sequences=True)(LSTM_out)
		# LSTM_out = LSTM(32,return_sequences=True)(LSTM_out)
		LSTM_out = LSTM(32)(LSTM_out)
		# LSTM_out = Dense(16, activation='softmax')(LSTM_out)
		LSTMs.append(LSTM_out)
	LSTMs = Add()(LSTMs)
	# out = Dense(len(sensors) * out_dim)(LSTMs)
	# out = Dense(int(len(sensors) * 8 / 2))(LSTMs)
	out = Dense(out_dim, activation='softmax')(LSTMs)
	model = Model(inp, out)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	print "\n\n\nInput --> LSTM(32)*6 --> Dense --> Output"
	return model


def lstm_model(x, y, sensors):
	out_dim = 1 if len(y.shape) == 1 else y.shape[1]
	inp = Input(shape=(x.shape[1], x.shape[2]), dtype='float')

	LSTM_out = LSTM(16, return_sequences=True)(inp)
	# LSTM_out = LSTM(8, return_sequences=True)(LSTM_out)
	LSTM_out = LSTM(16)(LSTM_out)

	out = Dense(out_dim, activation='softmax')(LSTM_out)
	model = Model(inp, out)
	model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
	print "\n\n\nInput --> LSTM(16,16) --> Dense --> Output"
	return model


def get_output_layer(model, layer_name):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	layer = layer_dict[layer_name]
	return layer.output


def train_model(model, x, y, filepath, load=False):
	# earlystop = EarlyStopping(monitor='loss')
	if load:
		model.load_weights(filepath)
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	hist = model.fit(x, y, epochs=40, batch_size=32, validation_split=0.1, callbacks=[checkpoint])
	loss, acc, val_loss, val_acc = hist.history['loss'], hist.history['acc'], hist.history['val_loss'], hist.history['val_acc']

	c = {"models/conv.hdf5": 'b', "models/lstm.hdf5": 'y', "models/hybrid.hdf5": 'g', "models/lstm6.hdf5": 'c'}

	plt.plot(range(len(loss)), loss, label='loss', color=c[filepath])
	plt.plot(range(len(acc)), acc, label='acc', color=c[filepath])
	plt.plot(range(len(val_loss)), val_loss, label='val_loss', color=c[filepath], marker='.')
	plt.plot(range(len(val_acc)), val_acc, label='val_acc', color=c[filepath], marker='.')
	model.save_weights(filepath)
	return model


def from_file(model, path):
	model.load_weights(path)
	return model


def test_model(model, x, y):
	pred_y = model.predict(x)
	if len(y.shape) == 1:
		pred_y = pred_y.round()
	else:
		pred_y = pred_y.argmax(axis=-1)
		y = [np.argmax(i) for i in y]
	print "\nConfusion Matrix:\n", confusion_matrix(y, pred_y)
	print "\nClassification Report:\n", classification_report(y, pred_y)
	return pred_y
