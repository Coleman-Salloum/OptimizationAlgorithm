"""Upper Confidence Bound Algorithm"""


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



# UCB Algorithm

# lists for storing numbers of pulls, empirical estimates of expected loss, lower confidence bounds for each arm
number_pulls = [0 for i in range(N)]
muhat = [0 for i in range(N)]
lcb = [0 for i in range(N)]

# variable for tracking total payout from arms chosen by UCB
ucb_payout = 0

# start by pulling each arm once and observing losses
for t in range(N):
	chosen_arm = t
	# update number of times arm has been pulled
	number_pulls[chosen_arm] += 1

	# observe arm payout and translate to loss
	loss = 1-payouts[chosen_arm][t]
	ucb_payout += payouts[chosen_arm][t]

	# initial empirical estimate of expected loss is just loss incurred on first pull
	muhat[chosen_arm] = loss

# now start pulling the arm with the smallest lower confidence bound at each round
for t in range(N,T):
	# calculate LCB for each arm
	for i in range(N):
		lcb[i] = muhat[i] - np.sqrt( 3 * np.log(t) / number_pulls[i] )

	# choose arm with smallest LCB
	chosen_arm = np.argmin(lcb)

	# observe arm payout and translate to loss
	loss = 1-payouts[chosen_arm][t]
	ucb_payout += payouts[chosen_arm][t]

	# update number of pulls, empirical estimate of expected loss for chosen arm
	number_pulls[chosen_arm] += 1
	muhat[chosen_arm] = ( muhat[chosen_arm] * (number_pulls[chosen_arm] - 1) + loss ) / number_pulls[chosen_arm]

print(muhat, ucb_payout)
