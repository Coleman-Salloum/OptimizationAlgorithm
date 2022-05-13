import math

# define relevant cost and revenue functions

def salvageCost(b):
	return 5 * b

def orderingCost(b):
	if b == 0:
		return 0
	else:
		return 4 + 2 * b

def holdingCost(b):
	return 1.5 * b

def revenue(b):
	return 8 * b

# problem parameters
M = 10   # warehouse capacity
N = 4   # number of decision epochs
probs = [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0, 0, 0]  # probability distribution for demand


S = [i for i in range(M+1)]  # state space
A = {}  # dictionary of action spaces for each state

# populate action spaces for each state
for s in S:
	A[s] = [i for i in range(M-s+1)]


p = {} # dictionary for storing transition probabilities
r = {} # dictionary for storing rewards

# compute and store transition probabilities
for s in S:
	for a in A[s]:
		for j in S:
			if j > s + a:
				p[j,s,a] = 0
			else:
				if j == 0:
					p[j,s,a] = sum( probs[i] for i in range(s+a,M+1) )
				else:
					p[j,s,a] = probs[s+a-j]

# store worst-case (smallest) reward for a single decision epoch, initializing as smallest terminal reward and updating below if needed.  
# Use later as a lower bound on overall total expected reward.
smallestReward = min(-salvageCost(s) for s in S)

# compute and store expected reward function values
for s in S:
	for a in A[s]:
		r[s,a] = -orderingCost(a) - holdingCost(s+a) + sum( (probs[k] * revenue(k)) for k in range(s+a+1) ) + revenue(s+a) * sum( probs[k] for k in range(s+a+1,M+1) )
		if smallestReward > r[s,a]:
			smallestReward = r[s,a]

dstar = {}  # dictionary for storing optimal decisions

# set up optimality equations and boundary condition
def ustar(t,s):
	# recursion
	if t < N:
		# placeholder for maximum expected reward u_t*(s), initialize using worst-case (smallest) total expected reward possible
		maxValue = N * smallestReward
		for a in A[s]:
			candidate = r[s,a] + sum( p[j,s,a] * ustar(t+1,j) for j in S )
			if candidate >= maxValue:
				maxValue = candidate
				dstar[t,s] = a
		return maxValue
	# boundary condition
	else:
		return -salvageCost(s)

v = {}
for s in S:
	v[s] = ustar(1,s)

print( v )
print( dstar )