__author__ = 'crisnv'

import math
import cv2

import numpy as np

#na img 83 nao apareceu nada...???

a = 0

def    findSymmetryAxis(inImgEl, inGTImg, dirVec, filenamePrefix):
#se com densidades diferentes, gaussiana,
#se identica, elipse no contorno

#algorithmo de  [Fitzgibbon95]
    global a
    a = a+1
    if a<200:
        return

    ret,thresh = cv2.threshold(inGTImg,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)


    print(len(contours))
    #pela descricao deveria ter so 1 cmp conexo, mas tem algumas imgs com mais
    #assert( len(contours) == 1)
    assert( len(contours) >0)




#   M = cv2.moments(cnt)
    #print M


#centroids:
#    cx = int(M['m10']/M['m00'])
#    cy = int(M['m01']/M['m00'])


#to fit an ellipse to an object. It returns the rotated rectangle in which the ellipse is inscribed.




    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    elp_ax1 = ellipse[1][0]
    elp_ax2 = ellipse[1][1]
    assert(elp_ax1 <= elp_ax2)

    for i in range(len(contours) -1):
        cnt = contours[i+1]
        ellipseAux = cv2.fitEllipse(cnt)
        elpAxis1Aux = ellipseAux[1][0]
        elpAxix2Aux = ellipseAux[1][1]

        if(elpAxis1Aux+elpAxix2Aux) > (elp_ax1+elp_ax2):
            ellipse = ellipseAux


    #mostrar em debug
    print(ellipse)
    inGTImgElp = cv2.cvtColor(inGTImg,  cv2.COLOR_GRAY2RGB)
    cv2.ellipse(inGTImgElp,ellipse,(0,255,0),2)


#    p1_x = int(elp_cx)
#    p1_y = int(elp_cy)


    elp_cx = ellipse[0][0]
    elp_cy = ellipse[0][1]
    elp_ax1 = ellipse[1][0]
    elp_ax2 = ellipse[1][1]
    elp_angl = ellipse[2]
    assert(elp_ax1 <= elp_ax2)

    #minAxis
    dx = math.cos(math.pi*elp_angl/180.0)*elp_ax1
    dy = math.sin(math.pi*elp_angl/180.0)*elp_ax1
    p1_x = int(elp_cx -dx)
    p1_y = int(elp_cy -dy)
    p2_x = int(elp_cx +dx)
    p2_y = int(elp_cy +dy)
    minAx_p1_x = p1_x
    minAx_p1_y = p1_y
    minAx_p2_x = p2_x
    minAx_p2_y = p2_y

    #debug:
    cv2.line(inImgEl, (p1_x, p1_y), (p2_x, p2_y), (0, 255,0), 2)
    cv2.line(inGTImgElp, (p1_x, p1_y), (p2_x, p2_y), (0, 255,0), 2)



    ### maxAxis
    dx = math.cos(math.pi*(elp_angl+90)/180.0)*elp_ax2
    dy = math.sin(math.pi*(elp_angl+90)/180.0)*elp_ax2
    p1_x = int(elp_cx -dx)
    p1_y = int(elp_cy -dy)
    p2_x = int(elp_cx +dx)
    p2_y = int(elp_cy +dy)
    maxAx_p1_x = p1_x
    maxAx_p1_y = p1_y
    maxAx_p2_x = p2_x
    maxAx_p2_y = p2_y


    #debug
    cv2.line(inImgEl, (p1_x, p1_y), (p2_x, p2_y), (0,0, 255), 2)
    cv2.line(inGTImgElp, (p1_x, p1_y), (p2_x, p2_y), (0,0, 255), 2)


    cv2.imshow('elipse', inImgEl)
    cv2.imshow('elipse GT', inGTImgElp)
    #cv2.waitKey(1000)

    outFileName  = dirVec['drawEllipses'] + filenamePrefix + '_elipse.png'
    print( outFileName )
    cv2.imwrite(outFileName, inImgEl)





    return

    #flip around min axix
    ptsIn = np.float32([[minAx_p1_x, minAx_p1_y],[maxAx_p1_x, maxAx_p1_y],[maxAx_p2_x, maxAx_p2_y]])
    ptsOut = np.float32([[minAx_p1_x, minAx_p1_y],[maxAx_p2_x, maxAx_p2_y],[maxAx_p1_x, maxAx_p1_y]])

    M = cv2.getAffineTransform(ptsIn,ptsOut)

    rows,cols,ch = inImgEl.shape
    inImgElFlip = cv2.warpAffine(inImgEl,M,(cols,rows))

    cv2.imshow('elipse fliped', inImgElFlip)
    cv2.waitKey(1)
