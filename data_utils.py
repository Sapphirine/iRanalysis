import random


K = 10
mu = 0
sigma = 0.1
path_to_csv = 'data/new_processed_file.csv'
output_csv = 'data/augmented_file.csv'
input_file = open(path_to_csv, 'r')
output_file = open(output_csv, 'w')

header = input_file.readline()
output_file.write(header)
header = header.strip().split(',')

for line in input_file:
    try:
        line_p = map(float, line.strip().split(','))
        new_lines = []

        for k in xrange(K):
            rnd = round(random.normalvariate(mu, sigma), 2)
            temp = [str(x * (1 + rnd)) for x in line_p]
            new_lines.append(','.join(temp) + '\n')

        output_file.write(''.join(new_lines))
    except:
        continue

output_file.close()
input_file.close()
