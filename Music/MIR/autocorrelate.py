import librosa

def ac(file):
    y,sr = librosa.load(file)
    i_min = 22050/2000
    i_max = 22050/50
    r = librosa.autocorrelate(y)
    R_B = r.copy()
    r[:int(i_min)]=0
    r[int(i_max):]=0
    print('r.shape',r.shape)
    argmax = r.argmax()
    print('r.argmax()',argmax)
    f0 = float(sr)/argmax
    return (f0,r,R_B,argmax,y)

def plot_ac(ac0,coef):
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(ac0[1][:coef])
    plt.subplot(4,1,2)
    plt.plot(ac0[2][:coef])
    plt.subplot(4,1,3)
    plt.plot(ac0[2])
    plt.subplot(4,1,4)
    plt.plot(ac0[-1])

f = 'C5s.wav'
ac0=ac(f)
plot_ac(ac0,1500)
