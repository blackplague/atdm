import matplotlib
matplotlib.use("Agg")
import os
import sys
from pylab import *
import numpy as np

# if not os.path.exists(sys.argv[1]) and os.path.isdir(sys.argv[1]):
#     raise Exception("Path not found: %s" % sys.argv[1])

# majordirlist = sys.argv[1]
# print majordirlist
# dirlist = os.listdir(majordirlist)


def calculateRMSE(directory):
    d = {}
    d['reconstruction'] = []
    d['training'] = []

    resultlist = []

    newlist = os.listdir(directory)

    for f in sorted(newlist):
        if 'reconstruction' in f:
            d['reconstruction'].append(directory+f)
        else:
            d['training'].append(directory+f)

    test = zip(d['reconstruction'], d['training'])

    print test

    test2 = [(np.reshape(imread(a),(1, 360000)), np.reshape(imread(b), (1, 360000))) for a, b in test]
    rmselist = {}

    for i in range (len(test2)):
        v1, v2 = test2[i]
        rmse = np.sqrt(np.dot((v1-v2), np.transpose((v1-v2))))[0][0]
        rmselist[i] = rmse

    rmselistint = rmselist.values()
    fname = directory.split('/')[-2]

    mydict = {}
    mydict[fname] = rmselistint

    resultlist.append(mydict)

    # filename = '%s.txt'

    # with open(filename, 'w') as f:
    #     f.write(rmseliststring + '\n' + str(rmsesum))

    return resultlist

# for directory in dirlist:
#     print "dir: %s" %directory
#     rlist = calculateRMSE(majordirlist+directory+'/')

# print rlist

# for item in rlist:
#     length = len(item.values())
#     if length == 0:
#         continue
#     rmsesum = sum(item.values())
#     print rmsesum
#     print item.keys()[0]
#     with open("rmse/%05d_" % rmsesum + item.keys()[0] + '.txt', 'w') as f:
#         f.write('\n'.join(map(str,item.values())) + '\n' + str(rmsesum))
