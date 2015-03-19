import csv

def read_data(filepath):
	'''
	Each row will be read into a dictionary with the format:
	{
		numeric: values of features A to I,
		catk: list representing vector for categorical feature K,
		catl: list representing vector for categorical feature L
	}
	'''
	data = []
	with open(filepath, 'r') as fin:
		rows = csv.reader(fin, delimiter=',')
		for row in rows:
			vals = list(map(int, row[:10]))
			kvals = list(map(int, row[10: 14]))
			lvals = list(map(int, row[14:]))
			allVals = dict(numeric=vals, catk=kvals, catl=lvals)
			data.append(allVals)
	return data

def read_data_into_rows(filepath):
    data = []
    with open(filepath, 'r') as fin:
        rows = list(map(lambda x: map(int, x), csv.reader(fin, delimiter=',')))
        return rows
#data = read_data("project_data/train.csv")