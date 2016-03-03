#
#
# Skin Lesion Analysis Towards Melanoma Detection
#
#   preparing input data


# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import random

from constants import *
from imgAugmentation import *
from colorAugmentation import *




###  usar http://www.tutorialspoint.com/python/python_multithreading.htm para paralelizar
def generateBase(dirInData,
                 dirInGroundTruth,
                 dirSaveData,
                 dirSaveGroundTruth
):
    dirVec = {'origScaled': None,
              'gtOrigScaled': None,
              'origCropped': None,
              'gtCropped': None,
              'drawEllipses': None
    }

    lastInImg = 0

    aux = os.walk(dirInData)
    for p in aux:


        print("p[0]->", p[0])
        # print("p[1]->", p[1])
        # print("p[2] (size:", len(p[2]), "->", p[2] )

        # w_minDim = 99999999
        # h_minDim = 99999999
        # w_maxDim = 0
        # h_maxDim = 0

        for fname in p[2]:

            idxPto = fname.find('.jpg')

            if idxPto >= 0:

                #openPictures:

                inDataFilename = dirInData + fname[:]
                inImg = cv2.imread(inDataFilename, cv2.CV_LOAD_IMAGE_COLOR)
                if inImg is None:
                    print('imagem nao encontrada')
                    continue

                filenamePrefix = fname[:idxPto]
                inGTFilename = dirInGroundTruth + filenamePrefix + '_Segmentation.png'
                inGTImg = cv2.imread(inGTFilename, cv2.CV_LOAD_IMAGE_UNCHANGED)
                if inGTImg is None:
                    print('imagem nao encontrada:  ', inGTFilename)
                    continue

                height, width, channels = inImg.shape
                gt_h, gt_w = inGTImg.shape

                assert (gt_h == height)
                assert (gt_w == width)

                dirVec['origScaled'] = dirSaveData + "origScaled_img/"
                if not os.path.exists(dirVec['origScaled']):
                    os.makedirs(dirVec['origScaled'])

                dirVec['gtOrigScaled'] = dirSaveGroundTruth + "origScaled_gt/"
                if not os.path.exists(dirVec['gtOrigScaled']):
                    os.makedirs(dirVec['gtOrigScaled'])

                dirVec['origCropped'] = dirSaveData + "origCropped_img/"
                if not os.path.exists(dirVec['origCropped']):
                    os.makedirs(dirVec['origCropped'])

                dirVec['gtCropped'] = dirSaveGroundTruth + "origCropped_gt/"
                if not os.path.exists(dirVec['gtCropped']):
                    os.makedirs(dirVec['gtCropped'])


                ##### dirs de debug:
                dirVec['drawEllipses'] = dirSaveData + "drawEllipses/"
                if not os.path.exists(dirVec['drawEllipses']):
                    os.makedirs(dirVec['drawEllipses'])



                    #    dirVec[ ] = dirSave + " /"
                    #    if not os.path.exists(dirVec[]):
                    #        os.makedirs(dirVec[ ])



                #retirar para uso real:
                inImg = cv2.resize(inImg, (0, 0), fx=0.5, fy=0.5)
                inGTImg = cv2.resize(inGTImg, (0, 0), fx=0.5, fy=0.5)

                createTrainingSamplesFromImg(inImg, inGTImg, dirVec, filenamePrefix)

                createColorAugmentationsByGama(inImg, inGTImg, dirVec, filenamePrefix)

                #if not isinstance( lastInImg, int ):
                #    createColorAugmentationsByGuideImage(inImg, lastInImg, inGTImg, dirVec, filenamePrefix)



                # height, width, channels = inImg.shape
                #
                # w_minDim = min(w_minDim, width)
                # w_maxDim = max(w_maxDim, width)
                #
                # h_minDim = min(h_minDim, height)
                # h_maxDim = max(h_maxDim, height)
                #
                # print("/n Min dims: ", w_minDim, h_minDim, "/n")
                # print(" Max dims: ", w_maxDim, h_maxDim, "/n")

                ##teste: do num de canais (channels)
                #print("w,h, channels: ", inGTImg.shape)
                #
                #debug:
                #                if(width<600) or (height<600 ):



                if (width > 4000) or (height > 2000):
                    cv2.imshow('original', inImg)
                    cv2.imshow('segmentacao', inGTImg)
                    cv2.waitKey(1)


                lastInImg = inImg


#('\n Min dims: ', 576, 542, '\n')
#('\n Max dims: ', 4288, 2848, '\n')

###########  end of generateBase










#produce new positive samples from random geometric transforms
#warp affine:
#   scale
#   translation
#   rotation
#   shear
#warp perspective:
#
#adding noise:
#   gaussian
#   sault and pepper
#   perlin noise
# http://libnoise.sourceforge.net/glossary/index.html#coherentnoisbetavariate








#dirVec eh o vetor com diferentes dir de saida
def cropFile(annotFile, dirVec, imgFilename, frameN):
    ### programar com leitura de video

    af = AnnotFile()
    #neste momento nao estou fazendo nada com occluded
    [pedestrians, pedestrians_fa, people, occluded] = af.loadAnnotFile(annotFile)

    inImg = cv2.imread(imgFilename, cv2.CV_LOAD_IMAGE_COLOR)
    if inImg is None:
        print('imagem nao encontrada')
        return

    frame = inImg  # cv2.cvtColor(inImg,cv2.COLOR_BGR2RGB)
    #cv2.imshow('image', frame)
    #cv2.waitKey(1)


    # mapa de pedestres
    pedestrianMap = None

    peopleMap = None
    peopleMapIntegral = None
    pedestrianMapIntegral = None

    numPessoasFrame = 0
    i = 0


    #######################
    #marca partes visiveis no mapa geral de pessoas
    #######################

    peopleMap = np.zeros_like(frame[:, :, 0])
    if len(people) > 0:
        for p in people:
            #print(p) cv2.waitKey(0)
            peopleMap[p[2]:p[2] + p[4], p[1]:p[1] + p[3]].fill(1)

    if len(occluded) > 0:
        for p in occluded:
            [lbl, x, y, w, h, visx, visy, visw, vish, occlu] = p
            if (occlu == 1):
                peopleMap[visy: visy + vish, visx: visx + visw].fill(1)
            else:
                peopleMap[y:y + h, x:x + w].fill(1)

    if len(pedestrians) > 0:
        for p in pedestrians:
            [lbl, x, y, w, h, visx, visy, visw, vish, occlu] = p
            if (occlu == 1):
                peopleMap[visy: visy + vish, visx: visx + visw].fill(1)
            else:
                peopleMap[y:y + h, x:x + w].fill(1)

    if len(pedestrians_fa) > 0:
        for p in pedestrians_fa:
            [lbl, x, y, w, h, visx, visy, visw, vish, occlu] = p
            if (occlu == 1):
                peopleMap[visy: visy + vish, visx: visx + visw].fill(1)
            else:
                peopleMap[y:y + h, x:x + w].fill(1)

    peopleMapIntegral = peopleMap.cumsum(1).cumsum(0)






    # constantes
    addX = (param['bBox'][0]) / 2
    addY = (param['bBox'][1] - param['pedBox'][1]) / 2
    height, width, channels = frame.shape


    # possui pessoas
    pedestrianMap = np.zeros_like(frame[:, :, 0])

    numPessoasFrame = 0
    numPessoasFrame += cropPos(frame, frameN, occluded, pedestrianMap, addX, addY, dirVec)
    numPessoasFrame += cropPos(frame, frameN, pedestrians_fa, pedestrianMap, addX, addY, dirVec)
    numPessoasFrame += cropPos(frame, frameN, pedestrians, pedestrianMap, addX, addY, dirVec)

    #criando imagem integral do mapa de pessoas
    if numPessoasFrame > 0:
        pedestrianMapIntegral = pedestrianMap.cumsum(1).cumsum(0)

        # cv2.imshow('ped map', pedestrianMapIntegral)
        #
        # cv2.imshow('people map', peopleMapIntegral)
        # cv2.waitKey(0)






    ###########RECORTE DE NEGATIVAS:
    #Proporcao entre negativos e positivos
    npr = param['negPosRatio']

    # intervalo de escalas
    ppMin = param['propPed']
    ppMax = width / param['bBox'][0]
    if (width / (1.0 * param['bBox'][0]) > height / (1.0 * param['bBox'][1])):
        ppMax = height / param['bBox'][1]


    #crisnv: * soma afim e persp +1 original agora cada negativa sera tbm tranformada
    negRange = npr * numPessoasFrame
    #* ( param['artificiais_afim'] + param['artificiais_persp'] + 1)


    #crisnv: i para indice da imaegm negativas
    i = 0

    for c in range(0, negRange):

        img = cropNegIoU(frame, pedestrianMapIntegral, peopleMapIntegral, ppMin, ppMax, width, height, pedestrians,
                         dirVec, frameN, i)

        if img is not None:
            #   cv2.imshow('neg', img)
            cv2.imwrite(dirVec['neg'] + "f" + str(frameN) + "_" + str(i) + ".jpg", img)

            flippedImg = cv2.flip(img, 1)
            cv2.imwrite(dirVec['neg'] + "f" + str(frameN) + "_" + str(i) + "_flipped.jpg", flippedImg)

            i = i + 1

    # print("Fim")
    return


generateBase(  #in dirs:
               '/Users/crisnv/Documents/databases/medicas/ISBI 2016/SkinLesion/ISBI2016_ISIC_Part1_Training_Data/',
               '/Users/crisnv/Documents/databases/medicas/ISBI 2016/SkinLesion/ISBI2016_ISIC_Part1_Training_GroundTruth/',
               #out dirs:
               '/Users/crisnv/Documents/databases/medicas/ISBI 2016/SkinLesion/augTree_P1_Training_Data/',
               '/Users/crisnv/Documents/databases/medicas/ISBI 2016/SkinLesion/augTree_P1_Training_GroundTruth/'
)
