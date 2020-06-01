import numpy as np
from numpy import array, sign, zeros
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
#from matplotlib.pyplot import plot,show,hold,grid

def get_packet(s):
    assert type(s) == np.ndarray

    q_u = zeros(s.shape)
    q_l = zeros(s.shape)

    #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.

    u_x = [0,]
    u_y = [s[0],]

    l_x = [0,]
    l_y = [s[0],]

    #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1,len(s)-1):
        if (sign(s[k]-s[k-1])==1) and (sign(s[k]-s[k+1])==1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k]-s[k-1])==-1) and ((sign(s[k]-s[k+1]))==-1):
            l_x.append(k)
            l_y.append(s[k])

    #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.

    u_x.append(len(s)-1)
    u_y.append(s[-1])

    l_x.append(len(s)-1)
    l_y.append(s[-1])

    #Fit suitable models to the data. Here I am using cubic splines, similarly to the MATLAB example given in the question.

    u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
    l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)

    #Evaluate each model over the domain of (s)
    for k in range(0,len(s)):
        q_u[k] = u_p(k)
        q_l[k] = l_p(k)

    #Plot everything
    #plot(s);hold(True);plot(q_u,'r');plot(q_l,'g');grid(True);show()
    plt.figure()
    plt.plot(q_u,'r')
    plt.plot(q_l,'b')
    plt.plot(s,'y')
    #plt.savefig('ul.png')
    plt.vlines(np.argmax(q_u),0,max(q_u))
    return q_l, q_u
