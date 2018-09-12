import numpy as np
import pandas as pd
import os


def parse_file(loc, desc):
	f = open(loc, 'r')
	if desc not in 'annbox':
		data = [d.strip().split("\t")[-1] for d in f.readlines()]
		return ",".join(data)
	elif desc == 'ann':
		line = f.read()
		return line.strip().replace("#FAULT=", "")
	else:
		return f.readlines()[0].strip().replace(" ", ",")


def get_files(sub):
	files = os.listdir(sub)
	rec = []
	for i in files:
		parse = i.split(".")
		run_wafer = parse[0]  # int(parse[0].split("_")[0]),int(parse[0].split("_")[1])
		if parse[1] not in 'annbox':
			desc = int(parse[1])
		else:
			desc = parse[1]
		rec.append([run_wafer, desc])
	return rec


def get_df(sub):
	rec = np.array(get_files(sub))
	descs = list(set(rec[:, 1]))
	descs.sort()
	df = pd.DataFrame(columns=['run', 'wafer'] + descs)
	unq = np.array([[int(i.split("_")[0]), int(i.split("_")[1])] for i in list(set(rec[:, 0]))])
	df['run'], df['wafer'] = unq[:, 0], unq[:, 1]
	for i in os.listdir(sub):
		parse = i.split(".")
		run, wafer = int(parse[0].split("_")[0]), int(parse[0].split("_")[1])
		row_i = df[(df.run == run) & (df.wafer == wafer)][parse[1]].index[0]
		df.loc[row_i, parse[1]] = parse_file(sub + '/' + i, parse[1])  # sub+'/'+i #
	return df
