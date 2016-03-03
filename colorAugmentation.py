
__author__ = 'crisnv'

import numpy as np
import cv2
from constants import *

import matplotlib.pyplot as plt


#FAZER O PCA DA ALEXNET
#color constancy ability

# Color transfers:
# http://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
# http://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

# baidu:
# http://www.wired.com/2015/02/science-one-agrees-color-dress/
# http://research.baidu.com/the-color-of-the-dress/


# fazer a look up para varios gamas...ok
#colocar epslons de fatores de brilho e contraste

tableNeg = 0
tablePos = 0


tableNegNorm = 0
tablePosNorm = 0

def createNormLUT():

    global tableNegNorm, tablePosNorm

    if tablePosNorm==0:
        tablePosNorm = []

        step = param['gamaStep']
        gamMax= param['gamaMax']

        #+.1 por conta de o for ser de intervalo aberto, se nao nao faz o ultimo
        for gamma in np.arange(1+step, gamMax+.1, step):
            tablePosNorm = tablePosNorm + [ np.array([((i / 255.0) ** gamma)
                                              for i in np.arange(0, 256)]).astype("float") ]



    if tableNegNorm==0:
        tableNegNorm = []

        step = param['gamaStep']
        gamMax= param['gamaMax']
        gamMin = param['gamaMin']

        nstep =  (gamMax-1)/step
        for gamma in np.arange(gamMin, 1, (1.0-gamMin)/nstep):
            tableNegNorm = tableNegNorm + [ np.array([((i / 255.0) ** gamma)
                                              for i in np.arange(0, 256)]).astype("float") ]

    assert(len(tableNegNorm)==len(tablePosNorm))




def createLUT():

    global tableNeg, tablePos

    if tablePos==0:
        tablePos = []

        step = param['gamaStep']
        gamMax= param['gamaMax']

        #+.1 por conta de o for ser de intervalo aberto, se nao nao faz o ultimo
        for gamma in np.arange(1+step, gamMax+.1, step):
            tablePos = tablePos + [ np.array([((i / 255.0) ** gamma) * 255
                                              for i in np.arange(0, 256)]).astype("uint8") ]



    if tableNeg==0:
        tableNeg = []

        step = param['gamaStep']
        gamMax= param['gamaMax']
        gamMin = param['gamaMin']

        nstep =  (gamMax-1)/step
        for gamma in np.arange(gamMin, 1, (1.0-gamMin)/nstep):
            tableNeg = tableNeg + [ np.array([((i / 255.0) ** gamma) * 255
                                              for i in np.arange(0, 256)]).astype("uint8") ]

    assert(len(tableNeg)==len(tablePos))









def adjust_gamma(image, gammaR=1.0, gammaG=1.0, gammaB=1.0):


    global table

    createLUT()
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
#    invGammaR = 1.0 / gamma
#    table = np.array([((i / 255.0) ** invGammaR) * 255
#	    	for i in np.arange(0, 256)]).astype("uint8")

    b,g,r = cv2.split(image)



    step = param['gamaStep']
    gamMax= param['gamaMax']
    gamMin = param['gamaMin']


    numStep = (gamMax-1)/step
    negStep = (1.0-gamMin)/numStep


    print("\n gammaB, gammaG, gammaR ", gammaB, gammaG, gammaR )

    #casos de gama = 1 deixa canal inalterado
    if gammaB<1:
        b = cv2.LUT(b, tableNeg[int((gammaB-gamMin)/negStep)])
    elif gammaB>1:
        b = cv2.LUT(b, tablePos[int((gammaB-1)/step)-1])

    if gammaG<1:
        g = cv2.LUT(g, tableNeg[int((gammaG-gamMin)/negStep)])
    elif gammaG>1:
        g = cv2.LUT(g, tablePos[int((gammaG-1)/step)-1])

    if gammaR<1:
        r = cv2.LUT(r, tableNeg[int((gammaR-gamMin)/negStep)])
    elif gammaR>1:
        r = cv2.LUT(r, tablePos[int((gammaR-1)/step)-1])

    #debug:
    #cv2.imshow('r',r)
    #cv2.imshow('g',g)
    #cv2.imshow('b',b)
    #cv2.waitKey(1)

    cv2.waitKey(0)

    img = cv2.merge([b,g,r])

    # apply gamma correction using the lookup table
    return img



def selectNormTable(gamma):

    step = param['gamaStep']
    gamMax= param['gamaMax']
    gamMin = param['gamaMin']
    numStep = (gamMax-1)/step
    negStep = (1.0-gamMin)/numStep

    t=0
    if gamma<1:
        t= tableNegNorm[int((gamma-gamMin)/negStep)]
    elif gamma>1:
        t = tablePosNorm[int((gamma-1)/step)-1]



    return t


def getLUTbySteps(tab1, tab2, pto):


    LUTaux = np.array([ 0
                        for i in np.arange(0, 256)]).astype("uint8")

    if not isinstance(tab1, int):
        for i in range(0, pto):
            #calculo real:
            # LUTaux[i] = pto* (i/(pto-1.0)) ** gamma
            #fast aproximation:
            LUTaux[i] = pto*tab1[(i/(1.0*pto)) * 255]
    else:
        for i in range (0,pto):
            LUTaux[i]=i


    if not isinstance( tab2, int ):
        for i in range (pto, 256):
            #fast aprox:
            LUTaux[i] = pto+(256-pto)*tab2[((i-pto)/(256.0-pto)) * 255]
            assert (LUTaux[i]<256)
    else:
        for i in range (pto, 256):
            LUTaux[i]=i



    return LUTaux

#ptos sao valores de 0 a 255 (se 256 faz a gama comum)
def adjust_gamma_bySteps(image, gammaR1=1.0, gammaG1=1.0, gammaB1=1.0, gammaR2=1.0, gammaG2=1.0, gammaB2=1.0, ptoR=64, ptoG=64, ptoB=64):

    global table

    createNormLUT()

    step = param['gamaStep']
    gamMax= param['gamaMax']
    gamMin = param['gamaMin']
    numStep = (gamMax-1)/step
    negStep = (1.0-gamMin)/numStep





    #
    print('gammaB, gammaG, gammaR ', gammaB1, gammaB2, gammaG1, gammaG2, gammaR1, gammaR2 )

    #casos de gama = 1 deixa canal inalterado
    tab1 = selectNormTable(gammaB1)
    tab2 = selectNormTable(gammaB2)

    LUTauxB = getLUTbySteps(tab1, tab2, ptoB)


    ####
    tab1 = selectNormTable(gammaG1)
    tab2 = selectNormTable(gammaG2)

    LUTauxG = getLUTbySteps(tab1, tab2, ptoG)


    ####
    tab1 = selectNormTable(gammaR1)
    tab2 = selectNormTable(gammaR2)

    LUTauxR = getLUTbySteps(tab1, tab2, ptoR)


    #debug:
    plt.clf()
    plt.plot(range(0,256),range(0,256),'y--',range(0,256), LUTauxR, 'r--', range(0,256), LUTauxG, 'bs', range(0,256), LUTauxB, 'g^')
    plt.ylabel('LUTaux')
    plt.draw()
    plt.show(block=False)

    b,g,r = cv2.split(image)

    r = cv2.LUT(r, LUTauxR)
    g = cv2.LUT(g, LUTauxG)
    b = cv2.LUT(b, LUTauxB)


    #debug:
    #cv2.imshow('r',r)
    #cv2.imshow('g',g)
    #cv2.imshow('b',b)
    #cv2.waitKey(1)

    img = cv2.merge([b,g,r])

    # apply gamma correction using the lookup table
    return img








def createColorAugmentationsByGama(original, inGTImg, dirVec, filenamePrefix):
    #cria uma com deslocamento positivo e outra com mesmo grau de deslocamento mas negativo
    #x^gama e x^(1/gama)

    #rand()
    #    gama = 2
    cv2.namedWindow("Images",0)


    step = param['gamaStep']
    gamMax= param['gamaMax']
    gamMin = param['gamaMin']

    numStep =  (gamMax-1)/step
    negStep = (1.0-gamMin)/numStep


    for gammaR in np.union1d(np.arange(gamMin, 1.0, negStep),  np.arange(1, gamMax+.1, step)):
        for gammaG in np.union1d(np.arange(gamMin, 1.0, negStep),  np.arange(1, gamMax+.1, step)):
           for gammaB in np.union1d(np.arange(gamMin, 1.0, negStep),   np.arange(1, gamMax+.1, step)):

                 adjusted = adjust_gamma(original,gammaR,gammaG,gammaB)
                 cv2.imshow("Images", np.hstack([original, adjusted]))
                 cv2.waitKey(100)



    #non blocking window
    #plt.ion()
    # loop over various values of gamma
    for gammaR1 in np.union1d(np.arange(gamMin, 1.0, negStep),  np.arange(1, gamMax+.1, step)):
        for gammaR2 in np.union1d(np.arange(gamMin, 1.0, negStep),  np.arange(1, gamMax+.1, step)):
            for gammaG1 in np.union1d(np.arange(gamMin, 1.0, negStep),   np.arange(1, gamMax+.1, step)):
                for gammaG2 in np.union1d(np.arange(gamMin, 1.0, negStep),   np.arange(1, gamMax+.1, step)):
                    for gammaB1 in np.union1d(np.arange(gamMin, 1.0, negStep),   np.arange(1, gamMax+.1, step)):
                        for gammaB2 in np.union1d(np.arange(gamMin, 1.0, negStep),   np.arange(1, gamMax+.1, step)):
                            # ignore when gamma is 1 (there will be no change to the image)
                            #if gamma == 1:
                            #    continue
                                # apply gamma correction and show the images
                            #gamma = gamma if gamma > 0 else 0.1

                            #adjusted = adjust_gamma(original,gammaR,gammaG,gammaB)
                            adjusted = adjust_gamma_bySteps(original,gammaR1,gammaG1,gammaB1,gammaR2,gammaG2,gammaB2)

                            cv2.imshow("Images 2", np.hstack([original, adjusted]))
                            cv2.waitKey(1)
                            #plt.close('all')







def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space
    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def color_transfer(source, target):
# """
# Transfers the color distribution from the source to the target
# image using the mean and standard deviations of the L*a*b*
# color space.
# This implementation is (loosely) based on to the "Color Transfer
# between Images" paper by Reinhard et al., 2001.
# Parameters:
# -------
# source: NumPy array
#     OpenCV image in BGR color space (the source image)
# target: NumPy array
#     OpenCV image in BGR color space (the target image)
# Returns:
# -------
# transfer: NumPy array
#     OpenCV image (w, h, 3) NumPy array (uint8)
# """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdTar / lStdSrc) * l
    a = (aStdTar / aStdSrc) * a
    b = (bStdTar / bStdSrc) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer



def createColorAugmentationsByGuideImage(original, guideImg, inGTImg, dirVec, filenamePrefix):


    guidedImg = color_transfer( guideImg, original)

    cv2.imshow('guide', guideImg)

    #cv2.imshow("Guided", np.hstack([original, guidedImg]))
    cv2.imshow("Guided", guidedImg)
    cv2.imshow("orig", original)


    cv2.waitKey(0)
