# melanoma:
# assimetria: cor, borda forma
#   borda irregular
#   borda assimetrica > mais risco
#   nao uniformidade da distribuicao de cor ( qtdd de cor ) > mais risco
#
#   cortes da lesao:  quadrantes diferentes
#   veu cinza azulado
#   tamanho > 0.6mil
#
#   veu: transparencia diferente do tom da pela acinzentado
#
#   regua do dermatoscopio
#
#   amostras de pele sadia para a segmentacao e para classificacao













__author__ = 'crisnv'


augmentationTree = (

    ('scale2fit', 'uniformScale_crop2fit', 'nonuniformCrop_Scale2fit'),
                      ('rotation', (0, 90, 180, 270)),
                      ('flip', ('none', 'vertical', 'horizontal', 'both' )),
                      ('flipByAxis', ('none', 'min', 'max', 'both' )),
# Invariant to illuminant of the scene:
                      ('colorRange', ('none', 'blur?', 'saturate', 'equalization', 'whiteBalancing', 'color casting')),
                      ('thinPlate', ('randomGrid', 'lensDistortion', 'radialOpening', 'radialClosing'))
                    )


param={
    'outImgDims':[64,128], # width X height

    'artificiais_afim': 10, #3,
    'artificiais_persp': 10, # 3,
#    'artificiais': param['artificiais_afim'] +     param['artificiais_persp'],
    'delta': 2,
    # num de negativas em bases so para criacao de negativas da Imagenet
    'negPerFile': 10,
    #borda para corte de negativos
    'cborder': 32,


    ####    color parameters
    'gamaMin': 0.75, #.5
    'gamaMax': 1.5,   #1
    'gamaStep':0.25,
#    brilho e contraste

}
