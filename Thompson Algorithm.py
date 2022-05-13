""" Thompson Algorithm"""


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

# Algorithm Start
# list for storing sampled probabilities of success for each arm
phat = [0 for i in range(N)]

# run algorithm K times
for k in range(K):
	
	# variable for tracking total payout from arms chosen by UCB
	ts_payout = 0

	# lists for storing alpha and beta parameters for each arm's beta distribution
	alpha = [1 for i in range(N)]
	beta = [1 for i in range(N)]

	# algorithm execution
	for t in range(T):
		# sample phat values from beta distributions
		for i in range(N):
			phat[i] = rng.beta(alpha[i],beta[i])

		# choose arm with largest phat value
		chosen_arm = np.argmax(phat)

		# observe arm payout
		reward = payouts[chosen_arm][t]
		ts_payout += reward

		# update beta distribution parameters for chosen arm
		if reward == 1:
			alpha[chosen_arm] += 1
		else:
			beta[chosen_arm] += 1

	ts_payouts.append(ts_payout)
	print(alpha, beta, ts_payout)

print(statistics.mean(ts_payouts), statistics.variance(ts_payouts))