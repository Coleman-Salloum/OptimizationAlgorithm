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


###############################################################

# Simple MAB algorithm with MWU as chosen full-information algorithm A

# probability of exploring
delta = 0.3

# pick a practical value for epsilon for MWU
epsilon = np.sqrt(np.log(N) / (T))

for k in range(K):

	# keep track of cumulative loss for Simple MAB
	simple_payout = 0

	# initial vector of arm selection probabilities
	x = [(1/N) for i in range(N)]

	# vector of weights for MWU
	weights = [1 for i in range(N)]

	# algorithm
	for t in range(T):
		# choose whether to explore or exploit
		explore_or_exploit = rng.binomial(1, delta)
		
		# exploration
		if explore_or_exploit == 1:
			# choose arm to pull uniformly at random
			chosen_arm = rng.integers(low=0, high=N)
			# observe loss--will be either 0 (no payout) or -1 (payout of $1)
			loss = 1-payouts[chosen_arm][t]
			simple_payout += payouts[chosen_arm][t]
			# update weight of chosen arm
			weights[chosen_arm] = weights[chosen_arm] * (1 - epsilon * loss)
			# update vector of selection probabilities
			W = sum(weights[i] for i in range(N))
			for i in range(N):
				x[i] = weights[i] / W

		# exploitation
		else:
			# choose arm to pull according to distribution x
			chosen_arm = rng.choice(N, p=x)
			# observe loss and update cumulative loss
			loss = 1-payouts[chosen_arm][t]
			simple_payout += payouts[chosen_arm][t]

	# display final vector of probabilities and cumulative reward
	print(x, simple_payout)

	# store cumulative reward for later analysis
	simple_payouts.append(simple_payout)

###############################################################



