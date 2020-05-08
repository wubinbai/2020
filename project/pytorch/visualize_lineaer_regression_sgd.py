%matplotlib inline

from fastai.basics import * 
from matplotlib import animation, rc



n=100
x = torch.ones(n,2) 
x[:,0].uniform_(-1.,1)
a = nn.Parameter(tensor(-1.,1))
y = x@a + torch.rand(n)

fig = plt.figure()
plt.scatter(x[:,0], y, c='orange')
with torch.no_grad():
  line, = plt.plot(x[:,0], x@a)
plt.close()

def animate(i):
    update()
    line.set_ydata(x@a)
    return line,

animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)
