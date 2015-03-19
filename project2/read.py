import csv

def read_x(filepath):
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
			vals = list(map(float, row[:10]))
			kvals = list(map(float, row[10: 14]))
			lvals = list(map(float, row[14:]))
			allVals = dict(numeric=vals, catk=kvals, catl=lvals)
			data.append(allVals)
	return data

def read_y(filepath):
    '''
    Each row will be read into a dict with the format:
    {
        Y: Numeric label in range 1-7
        Z: Numeric label in range 0-2
    }
    '''
    data = []
    with open(filepath, 'r') as fin:
        rows = csv.reader(fin, delimiter=',')
        for row in rows:
            rowVals = dict(Y=int(row[0]), Z=int(row[1]))
            data.append(rowVals)
    return data

#data = read_data("project_data/train.csv")
