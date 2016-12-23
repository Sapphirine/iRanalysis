import random

K = 4
mu = 0
sigma = 0.01
path_to_csv = 'data/new_processed_file.csv'
output_csv = 'data/augmented_file_2.csv'
input_file = open(path_to_csv, 'r')
output_file = open(output_csv, 'w')

header = input_file.readline()
output_file.write(header)
header = header.strip().split(',')

for line in input_file:
    try:
        line_p = map(float, line.strip().split(','))
        new_lines = []

        rnd = 0
        for k in xrange(K):
            temp = [round(x * (1 + rnd), 2) for x in line_p]
            temp[6] = max(2**6, temp[6])
            temp = map(str, temp)
            new_lines.append(','.join(temp) + '\n')
            rnd = round(random.normalvariate(mu, sigma), 2)

        output_file.write(''.join(new_lines))
    except:
        continue

output_file.close()
input_file.close()
