# import matplotlib as mpl
# mpl.use('agg')
import os
# import mdp
from mdp.nodes import RBMWithLabelsNode
import numpy as np
# from numpy import *
from pylab import *
# import random
import readDataset
# reload(readDataset)

class Random():
    pass

class Sequence():
    pass

def readAllFiles(path):
    """
    No sanity check, just reads in every file
    """
    l = []

    listing = os.listdir(path)
    for infile in listing:
        im = imread(path+infile)
        im = np.reshape(im, (1, 256))
        l.append(im)

    return l

def runRBM(training='sequence', debug=False):

    # Want to use class to identify type instead:
    # if training == 'sequence':
    #     training = Sequence()
    # elif training == 'random':
    #     training == Random()
    # else:
    #     raise Exception("Unknown training pattern: %s" % training)

    # digits = [1,8]
    # trainingimagesdict = {}
    # traininglabelsdict = {}

    # print "Loading following training digits from MNIST: %s" % " ".join(map(str,digits))
    # for digit in digits:
    #     trainingimages, traininglabels =  \
    #         readDataset.read([digit], dataset='training', path='./images/mnist')
    #     trainingimagesdict[digit] = np.array(trainingimages)
    #     traininglabelsdict[digit] = np.array(traininglabels)

    # print "Done loading training images."

    # testingimagesdict = {}
    # testinglabelsdict = {}

    # print "Loading following testing digits from MNIST: %s" % " ".join(map(str,digits))

    # for digit in digits:
    #     testingimages, testinglabels =  \
    #         readDataset.read([digit], dataset='testing', path='./images/mnist')
    #     testingimagesdict[digit] = np.matrix(testingimages)
    #     testinglabelsdict[digit] = np.matrix(testinglabels)

    # print "Done loading testing images."

    # # Detect the dimension of what we are operating on
    # print "Detecting dimensions of the problem:"
    # if trainingimagesdict[digits[0]].shape[1]:
    #     visibledim = trainingimagesdict[digits[0]].shape[1]
    #     labelsdim   = traininglabelsdict[digits[0]].shape[1]
    #     imagedim   = int(np.sqrt(trainingimagesdict[digits[0]].shape[1]))
    #     if debug:
    #         print "Visible dimension: %s" % visibledim
    #         print "Label dimension:   %s" % labelsdim
    #         print "Image dimension:   %s" % imagedim
    # else:
    #     raise Exception("Unable to determine dimensions of the problem")

    numberoftrainingimages = 30

    chosen = {}
    # # if isinstance(training, Sequence):
    if training == 'sequence':
        trainingmatrix = np.array(np.zeros((numberoftrainingimages*len(digits), visibledim), dtype=np.float32))
        traininglabels = np.array(np.zeros((numberoftrainingimages*len(digits), labelsdim), dtype=np.int32))
        for i in range(len(digits)):
            trainingmatrix[range(i*numberoftrainingimages,(i+1)*numberoftrainingimages), :] = np.array(trainingimagesdict[digits[i]][range(numberoftrainingimages), :]/255.0)
            traininglabels[range(i*numberoftrainingimages,(i+1)*numberoftrainingimages), :] = np.array(traininglabelsdict[digits[i]][range(numberoftrainingimages), :])
            chosen[digits[i]] = []
            chosen[digits[i]] = range(numberoftrainingimages)
        if debug:
            print "Sequence Traininglist:"
            for d in digits:
                print "Chosen[%s]: %s" % (d, chosen[d])

    for i in range(trainingmatrix.shape[0]):
        print trainingmatrix[i,:]*255

    print

    hiddendim          = 784
    trainingiterations =  25
    cd_steps           =   1
    # rbm = mdp.nodes.RBMWithLabelsNode(hiddendim, labelsdim, visibledim, \
    #                                       dtype=np.float32)

    rbm = RBMWithLabelsNode(hiddendim, labelsdim, visibledim, \
                                          dtype=np.float32)


    # We should perhaps choose some way to make early stopping. For now
    # we just to it for n = trainingiterations
    print "Starting to train rbm for %s number of iterations" % trainingiterations

    if debug:
        print "Training matrix dimensions: %s" % " x ".join((map(str,trainingmatrix.shape)))
    print "Training iteration:"

    for i in range(trainingiterations):
        # if debug:
            # print " %s" % str(i+1),
            # print trainingmatrix.shape
            # print traininglabels.shape
        rbm.train(trainingmatrix[:,:], \
                      traininglabels[:,:], \
                      n_updates=cd_steps, epsilon=0.1, \
                      decay=0.0, momentum=0.0, verbose=False)

    # Load a test image
    testimage = np.array(np.zeros((1, visibledim), dtype=np.float32))
    # testimage[:,:] = np.array(testingimagesdict[digits[0]][0,:]/255.0)
    testimage[:,:] = np.array(trainingmatrix[0,:]/255.0)
    testlabel = np.array(np.zeros((1, 1), dtype=np.int32))
    # testlabel[:,:] = np.array(testinglabelsdict[digits[0]][0,:])
    testlabel[:,:] = np.array(traininglabels[0,:])

    print "Testlabel: %s" % testlabel

    h_probs, h_sample = rbm.sample_h(testimage, testlabel)
    # print h
    v_probs, l_probs, v_sample, l_sample = rbm.sample_v(h_sample)

    print "Energy: %s" % rbm.energy(v_sample, h_sample, testlabel)

    # prettyPrintProbabilities(v_prob, imagedim, width=5)
    # prettyPrintImage(v_sample, imagedim, width=2)

    v_sample = np.reshape(v_sample, (imagedim, imagedim))
    v_probs   = np.reshape(v_probs, (imagedim, imagedim))

    fig = figure(figsize=(1,1))
    imshow(np.reshape(np.array(testimage), (imagedim,imagedim)))
    fig.savefig('images/sample/test.png')


    fig = figure(figsize=(1,1))
    imshow(v_probs)
    fig.savefig('images/sample/prob_v.png')

    fig = figure(figsize=(1,1))
    imshow(v_sample)
    fig.savefig('images/sample/image_v.png')

    print "Prob_l: %s l: %s" % (l_probs, l_sample)


####### FOR RANDOM TRAINING DATA #######
    # # elif isinstance(training, Random):
    # # elif training == 'random':
    # #     for digit in digits:
    # #         chosen[digit] = []
    # #         trainingdict[digit] = []
    # #         while len(chosen[digit]) < numberoftrainingimages:
    # #             # if debug:
    # #             #     print "Range max: %s" % trainingimagesdict[digit].shape[0]
    # #             trainimage = random.randint(0, trainingimagesdict[digit].shape[0])
    # #             print trainimage
    # #             # If we already have the image, skip rest, draw new image.
    # #             if trainimage in chosen[digit]:
    # #                 print "Image: %s already chosen, redraw new." % trainimage
    # #                 continue
    # #             # We did not have the image, append to training list and update
    # #             # already chosen.
    # #             trainingdict[digit].append(trainingimagesdict[digit][trainimage,:])
    # #             chosen[digit].append(trainimage)
    # #     if debug:
    # #         print "Random Traininglist:"
    # #         print "Chosen: %s" % chosen
    # #         # print traininglist
    # #         print "len: %s" % len(trainingdict)
    # #         for tl in trainingdict:
    # #             print "len[%s]: %s" % (tl, len(trainingdict[tl]))
    # else:
    #     raise Exception("Unknown case")


    # testimage = testingimagesdict[digits[0]][0,:]/255.0
    # testlabel = testinglabelsdict[digits[0]][0,:]

    # # print testimage
    # # print testlabel

    # prob_h, h = rbm.sample_h(np.array(testimage), np.array(testlabel))
    # print h
    # v_prob, probs_l, v_sample, l = rbm.sample_v(h)

    # imdim = 28
    # prettyPrintProbabilities(v_prob, imdim, width=5)
    # prettyPrintImage(v_sample, imdim, width=2)

    # v_sample = np.reshape(v_sample, (imdim, imdim))
    # v_prob   = np.reshape(v_prob, (imdim, imdim))

    # fig = figure(figsize=(imdim,imdim))
    # imshow(np.reshape(np.array(testimage), (imdim,imdim)))
    # fig.savefig('images/sample/test.png')


    # fig = figure(figsize=(imdim,imdim))
    # imshow(v_prob)
    # fig.savefig('images/sample/prob_v.png')

    # fig = figure(figsize=(imdim,imdim))
    # imshow(v_sample)
    # fig.savefig('images/sample/image_v.png')

    # print "Prob_l: %s l: %s" % (probs_l, l)

    ################## WORKING DO NOT TOUCH (WITHOUT LABELS) ####################
    # rbm = mdp.nodes.RBMNode(hiddendim, visibledim,dtype=np.float32)

    # trainingiterations=10
    # cd_steps = 1

    # # We should perhaps choose some way to make early stopping. For now
    # # we just to it for n = trainingiterations
    # for i in range(trainingiterations):
    #     if debug:
    #         print "Training iteration: ", i+1
    #     for digit in digits:
    #         for i in range(trainingmatrixdict[digit].shape[0]):
    #             # trainingmatrixdict[digit]
    #             # print trainingmatrixdict[digit].ndim
    #             # print trainingmatrixdict[digit].shape
    #             rbm.train(np.array(trainingmatrixdict[digit][i,:]), \
    #                           n_updates=cd_steps, epsilon=0.1, \
    #                           decay=0.0, momentum=0.0, verbose=False)


    # print dir(rbm)
    # print rbm.energy() # Requires a hidden and visible configurations

    # numofsamples = 5

    # for i in range(numofsamples):
    #     randomData = generateRandom(hiddendim)
    #     v_prob, v_sample = rbm.sample_v(np.array(randomData))

    #     v_sample = np.reshape(v_sample, (imdim, imdim))
    #     v_prob   = np.reshape(v_prob, (imdim, imdim))
    #     fig = figure(figsize=(16,16))
    #     imshow(v_sample)
    #     fig.savefig('images/sample/im_sample%02d.png' % i)
    #     imshow(v_prob)
    #     fig.savefig('images/sample/prob_sample%02d.png' % i)

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
