"""EXP3 Algorthim"""

import math
import numpy as np
import csv
import statistics

# initialize random number generator
rng = np.random.default_rng(0)

# numbers of arms, rounds
N = 5
T = 1000

# number of times to run each randomized algorithm, and lists for storing total payouts from each run
K = 100
simple_payouts = []
exp3_payouts = []
ts_payouts = []

# lists of 1000 payouts (losses) for each arm
payouts = [ [] for i in range(N) ]

# read in arm payouts from csv file
with open('payouts.csv', newline='') as csvfile:
	csv_reader = csv.reader(csvfile, delimiter=',')
	next(csv_reader) # skip row with column headers
	# fill out lists with payouts
	for row in csv_reader:
		payouts[0].append(int(row[0]))
		payouts[1].append(int(row[1]))
		payouts[2].append(int(row[2]))
		payouts[3].append(int(row[3]))
		payouts[4].append(int(row[4]))



# EXP# Start

# run algorithm K times
for k in range(K):
	# keep track of cumulative loss for EXP3
	exp3_payout = 0

	# initial vector of arm selection probabilities
	x = [(1/N) for i in range(N)]

	# pick an appropriate value for epsilon
	epsilon = np.sqrt(np.log(N) / (N*T))

	# algorithm
	for t in range(T):
		# placeholder vector y used to update x
		y = x.copy()
		# choose arm to pull according to distribution x
		chosen_arm = rng.choice(N, p=x)
		# observe loss and update cumulative loss
		loss = 1-payouts[chosen_arm][t]
		exp3_payout += payouts[chosen_arm][t]
		# calculate y-value for chosen arm
		y[chosen_arm] = x[chosen_arm] * np.exp(-epsilon * loss / x[chosen_arm])
		# update vector of selection probabilities
		ysum = sum(y[i] for i in range(N))
		for i in range(N):
			x[i] = y[i] / ysum

	# display final vector of probabilities and cumulative reward
	print(x, exp3_payout)

	# store cumulative reward for later analysis
	exp3_payouts.append(exp3_payout)


# calculate mean and variance of 100 cumulative rewards for each algorithm
# use statistics.mean(list) and statistics.variance(list)
# print(statistics.mean(simple_payouts), statistics.variance(simple_payouts))

print(statistics.mean(exp3_payouts), statistics.variance(exp3_payouts))