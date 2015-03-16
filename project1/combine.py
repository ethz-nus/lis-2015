import numpy as np
def average_data(*args):
	items = []
	for filename in args:
		with open(filename, 'r') as item:
			items.append(list(map(lambda x: float(x), item)))
	if items:
		combined = np.array([sum([item[i]/len(items) for item in items]) for i in range(len(items[0]))])
		np.savetxt('result_validate.txt-combined', combined)
average_data('result_validate.txt', 'result_validate.txt-0.419867', 'result_validate.txt-0.419786', 'result_validate.txt-0.417175')