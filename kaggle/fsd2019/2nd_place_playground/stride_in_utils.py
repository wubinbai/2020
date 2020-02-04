stride=8

for i in range(10):
    s = i * stride
    wl = max(0,int(s-stride))
    wr = int(s + 1.5*stride)
    print('[',wl,',',wr,']')

