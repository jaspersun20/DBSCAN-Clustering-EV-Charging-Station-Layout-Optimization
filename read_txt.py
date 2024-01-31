import numpy as np

def str2xylst(loc_str):
    x,y = loc_str.split(' ')
    return float(x), float(y)

f=open('location.txt', encoding='gbk')

for line in f:
    id, loc=line.strip().split(';')[0], line.strip().split(';')[1][11:].strip('()').split(',')
    if len(loc) > 1:
        lst_x, lst_y =[], []
        for i in range(len(loc)):
            x, y = str2xylst(loc[i])
            lst_x.append(x)
            lst_y.append(y)
