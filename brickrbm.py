import matplotlib
matplotlib.use("Agg")
import os

from mdp.nodes import RBMNode
import numpy as np
from numpy import *
from pylab import *
import random

def readAllFiles(path):
    """
    No sanity check, just reads in every file
    """

    listing = os.listdir(path)
    trainingset = np.array(zeros((len(listing), 784), dtype=np.float32))

    counter = 0
    # for infile in listing:
    for i in range(len(listing)):
        print "Reading file %s" % listing[i] #path+infile
        print "listing i: %s" % path+listing[i]
        im = imread(path+listing[i])#+infile)
        im = np.reshape(im, (1, 784))
        trainingset[counter,:] = im
        counter += 1

    return trainingset

def runRBM():

    trainingset = np.array(readAllFiles('images/bricktraining/'))

    # print type(trainingset)
    # print trainingset.shape
    # print trainingset[1]
    # print trainingset

    # print "len: %s" % len(trainingset)
    # print "size: %s" % size(trainingset)

    # print "len: %s" % len(trainingset[0])
    # print "size: %s" % size(trainingset[0])

    # filename="images/train/stripe.pgm"
    # im = readImage(filename)

    # print size(trainingset[0])

    if size(trainingset[0]):
        visibledim = size(trainingset[0])
        imagedim = int(np.sqrt(visibledim))
    else:
        raise Exception("The trainingset contains no data, unable to obtain visible dimensions of the rbm.")

    hiddendim  = [256]
    trainingiterations = [3000]
    learningrate = [0.05]
    mom = [0.0] #, 0.2, 0.4, 0.5]
    dec = [0.00001]
    cdsteps = [1]
    reconstructionname = "brick"

    rbm = RBMNode(256, visibledim, dtype=np.float32)

    for i in range(trainingiterations[0]):
        rbm.train(trainingset, n_updates=1, epsilon=0.05, decay=dec[0], momentum=0.0, verbose=False)
        print i
                            # for i in range(trainingset.shape[0]):
                            #     train = np.array(zeros((1,visibledim), dtype=np.float32))
                            #     train[0,:] = trainingset[i,:]
                            #     h_prob, h_sample = rbm.sample_h(train)
                            #     v_prob, v_sample = rbm.sample_v(h_sample)
                            #     train = np.reshape(train, (imagedim, imagedim))
                            #     v_prob   = np.reshape(v_prob, (imagedim, imagedim))
                            #     fig = figure(figsize=(3,3))
                            #     infotuple = (reconstructionname, h, t, c, l, m, d, i)
                            #     if not os.path.exists('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/' % infotuple[0:7]):
                            #         os.mkdir('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/' % infotuple[0:7])
                            #     imshow(train, cmap=cm.gray)
                            #     fig.savefig('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/%02d_training.png' % infotuple)
                            #     imshow(v_prob, cmap=cm.gray)
                            #     fig.savefig('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/%02d_reconstruction.png' % infotuple)
                            # del rbm



    # # Try to make it come up with new images by gibbs sampling.
    gibbsstep = 15000
    sampleevery = 500

    v_sample = np.array(zeros((1,visibledim), dtype=np.float32))
    v_sample[0,:] = trainingset[4,:]

    for i in range(gibbsstep):
        if i % sampleevery == 0 and i != 0:
            fig = figure(figsize=(3,3))
            v_image = np.array(zeros((imagedim,imagedim), dtype=np.float32))
            v_imageprob = np.array(zeros((imagedim,imagedim), dtype=np.float32))
            v_image = np.reshape(v_sample, (imagedim, imagedim))
            v_imageprob = np.reshape(v_prob, (imagedim, imagedim))
            imshow(v_image, cmap=cm.gray)
            fig.savefig('images/gibbs_step_%04d.png' % i)
            imshow(v_imageprob, cmap=cm.gray)
            fig.savefig('images/gibbs_prob_%04d.png' % i)

        h_prob, h_sample = rbm.sample_h(np.array(v_sample))
        v_prob, v_sample = rbm.sample_v(np.array(h_sample))


    ##### WORKING ####

    #     rbm = RBMNode(hiddendim, visibledim, dtype=np.float32)


    # trainingiterations = 1500
    # learningrate = 0.1
    # m = 0.0
    # d = 0.0
    # cdsteps = 1

    # for i in range(trainingiterations):
    #     print "Iteration: %s" % str(i)
    #     rbm.train(trainingset, n_updates=cdsteps, epsilon=learningrate, decay=d, momentum=m, verbose=True)

    # reconstructionname = "brick"

    # for i in range(trainingset.shape[0]):
    #     train = np.array(zeros((1,visibledim), dtype=np.float32))
    #     train[0,:] = trainingset[i,:]
    #     h_prob, h_sample = rbm.sample_h(train)
    #     v_prob, v_sample = rbm.sample_v(h_sample)
    #     train = np.reshape(train, (imagedim, imagedim))
    #     v_prob   = np.reshape(v_prob, (imagedim, imagedim))
    #     fig = figure(figsize=(3,3))
    #     infotuple = (reconstructionname, hiddendim, trainingiterations, cdsteps, learningrate, m, d, i)
    #     if not os.path.exists('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/' % infotuple[0:7]):
    #         os.mkdir('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/' % infotuple[0:7])
    #     imshow(train, cmap=cm.gray)
    #     fig.savefig('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/%02d_training.png' % infotuple)
    #     imshow(v_prob, cmap=cm.gray)
    #     fig.savefig('images/reconstruction_%s_%s_%s_%s_%s_%s_%s/%02d_reconstruction.png' % infotuple)


    # numofsamples = 40

    # for i in range(numofsamples):
    #     randomData = generateRandom(hiddendim)
    #     v_prob, v_sample = rbm.sample_v(np.array(randomData))

    #     v_sample = np.reshape(v_sample, (imagedim, imagedim))
    #     v_prob   = np.reshape(v_prob,   (imagedim, imagedim))
    #     fig = figure(figsize=(3,3))
    #     # imshow(v_sample, cmap=cm.gray)
    #     # fig.savefig('images/sample/im_sample%02d.png' % i)
    #     imshow(v_prob, cmap=cm.gray)
    #     fig.savefig('images/sample/prob_sample%02d.png' % i)

    # # Try to make it come up with new images by gibbs sampling.

    # gibbsstep = 5000
    # sampleevery = 500

    # initsample = generateRandom(hiddendim)

    # for i in range(gibbsstep):
    #     v_prob, v_sample = rbm.sample_v(np.array(randomData))



    ################## WORKING DO NOT TOUCH ####################

    # randomData = generateRandom(hiddendim)
    # print "Length random: %s" % len(randomData)
    # print "Size random: %s" % size(randomData)
    # print "Random data:"
    # prettyPrintImage( randomData, imdim )

    # v_prob, v_sample = rbm.sample_v(np.array(randomData))

    # print
    # print "prettyPrintProbabilities(v_prob):"
    # prettyPrintProbabilities(v_prob, imdim)
    # print
    # print "prettyPrintImage(v_sample):"
    # prettyPrintImage(v_sample, imdim)

    # v_sample =np.reshape(v_sample, (imdim, imdim))

    # fig = figure(figsize=(16,16))
    # imshow(v_sample)
    # fig.savefig('images/sample/test2.png')


def prettyPrintProbabilities(problist, imdim, width=6):
    for i in range(size(problist)):
        if i != 0 and i % imdim == 0:
            print
        print "{0:{width}.1%} ".format(float(problist[0][i]), width=width),
    print



def prettyPrintImage(imagedata, imdim, width=2):
    for i in range(size(imagedata)):
        if i != 0 and i % imdim == 0:
            print
        print "{0:{width}}".format(int(imagedata[0][i]), width=width),
    print

def generateRandom(size=256):

    l = []
    for i in range(size):
        l.append(random.randrange(0,2))

    return [np.array(l)]
