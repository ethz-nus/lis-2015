import sys
def fix(inf, ref):
	infile = open(inf, 'r')
	refile = open(ref, 'r')
	items = []
	reff = list(refile)
	i = 0
	for line in infile:
		if 'nan' in line:
			items.append(reff[i])
		else:
			items.append(line)
		i += 1
	with open('fixed'+inf, 'w') as f:
		for item in items:
			f.write(item)
	
f1 = sys.argv[1]
f2 = sys.argv[2]
fix(f1, f2)