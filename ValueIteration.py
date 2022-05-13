import math

# set lambda, epsilon and some upper limit on the number of iterations
discount = 0.9
N = 1000
epsilon = 0.01

# dictionary for storing vectors from every iteration
v = {}

S = ["H", "L"]  # state space
A = ["C", "D"]  # dictionary of action spaces for each state

p = {} # dictionary for storing transition probabilities
r = {} # dictionary for storing rewards

# define transition probs
p["H", "H", "C"] = 0.8
p["L", "H", "C"] = 0.2
p["H", "H", "D"] = 0.4
p["L", "H", "D"] = 0.6
p["H", "L", "C"] = 0.7
p["L", "L", "C"] = 0.3
p["H", "L", "D"] = 0.5
p["L", "L", "D"] = 0.5

# expected immediate rewards
r["H", "C"] = 35
r["H", "D"] = 25
r["L", "C"] = 5
r["L", "D"] = 10

# initialize v0
v[0, "H"] = 0
v[0, "L"] = 0

upperBound = (epsilon * (1 - discount) ) / (2 * discount)

# binary for using Gauss-Seidel or standard value iteration
GS = 1

# value iteration
for n in range(1,N):
	if GS == 0:
		for s in S:
			v[n,s] = max( (r[s,a] + sum( (discount * p[j,s,a] * v[n-1,j]) for j in S )) for a in A)
	else:
		v[n,"H"] = max( (r["H",a] + sum( (discount * p[j,"H",a] * v[n-1,j]) for j in S )) for a in A)
		v[n,"L"] = max( (r["L",a] + discount * p["H","L",a] * v[n,"H"] + discount * p["L","L",a] * v[n-1,"L"] ) for a in A)
	print( n, "&", round(v[n,"H"],5), "&", round(v[n,"L"],5), "\\\\" )
	if ( abs(v[n,"H"] - v[n-1,"H"]) < upperBound and abs(v[n,"L"] - v[n-1,"L"]) < upperBound ):
		last = n
		break

# find optimal policy
d = {}

if ( r["H","C"] + sum( discount * p[j,"H","C"] * v[last,j] for j in S ) > r["H","D"] + sum( discount * p[j,"H","D"] * v[last,j] for j in S ) ):
	d["H"] = "C"
else:
	d["H"] = "D"

if ( r["L","C"] + sum( discount * p[j,"L","C"] * v[last,j] for j in S ) > r["L","D"] + sum( discount * p[j,"L","D"] * v[last,j] for j in S ) ):
	d["L"] = "C"
else:
	d["L"] = "D"

print( d )