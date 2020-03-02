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
    # algo my 
    localmax = int(i_min)
    temp = localmax+1
    start = True
    while start:
        if r[temp] < r[localmax]:
            decrease = True
        else:
            decrease = False
        if decrease == False:
            increase = True
            break
        localmax+=1
        temp+=1
    while increase:
        localmax+=1
        temp+=1
        if r[temp] > r[localmax]:
            increase = True
        else:
            increase = False
        if increase == False:
            break

    print('r.argmax()',argmax)
    f0 = float(sr)/argmax
    return ((f0,localmax),r,R_B,argmax,y)

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

f2 = 'A4C5s.wav'
ac2 = ac(f2)
plot_ac(ac2,1500)
