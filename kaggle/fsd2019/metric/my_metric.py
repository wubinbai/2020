from lwlrap import *

def get_lwlrap(a,b):
    c,d = calculate_per_class_lwlrap(a,b)
    sc = (c*d).sum()
    print('lwlrap: ', sc)
    return sc


y_true = np.array([[1, 0, 0,0,0], [1, 1, 0,0,0]])
y_score = np.array([[0.7, 0.3, 0.2,0.5,0.6],[0.8, 0.7, 0.9,0.508451,0.400515]])

temp = get_lwlrap(y_true,y_score)
