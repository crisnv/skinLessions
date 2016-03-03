__author__ = 'crisnv'



from SymmetryTransforms import *


def createTrainingSamplesFromImg(inImg, inGTImg, dirVec, filenamePrefix):





    findSymmetryAxis(inImg, inGTImg, dirVec, filenamePrefix)
#

#    gt_h, gt_w = inGTImg.shape
#    for i in range(gt_h):
#        for j in range(gt_w):

#            if(inGTImg[i,j]>0):
#                #calcula as distribuicoes das posicoes de x e y

