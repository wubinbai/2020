import random

N = 1000000
D = list(range(N))
Dp = []
for i in range(N):
    Dp.append(random.choice(D))

Dp = set(Dp)
D = set(D)
diff = D - Dp
print(len(diff)/len(D),1/np.e)
