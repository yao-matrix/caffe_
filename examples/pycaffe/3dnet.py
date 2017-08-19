import caffe
from caffe import layers as L, params as P, to_proto

def mynet():
    data, label = L.DummyData(shape=[dict(dim=[8, 1, 28, 28]),
                                     dict(dim=[8, 1, 1, 1])],
                              transform_param=dict(scale=1./255), ntop=2)

    # CAFFE = 1
    # MKL2017 = 3
    kwargs = {'engine': 3}
    conv1 = L.Convolution(data, kernel_size=[3, 4, 5], num_output=3, pad=[1, 2, 3])
    bn1 = L.BatchNorm(conv1, **kwargs)
    relu1 = L.ReLU(bn1, **kwargs)

    convargs = {'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=2)],
                'convolution_param': dict(num_output=64, 
                                          kernel_size=2,
                                          stride=2,
                                          engine=P.Convolution.CAFFE,
                                          bias_filler=dict(type='constant', value=0),
                                          weight_filler=dict(type='xavier'))
               }
    deconv1 = L.Deconvolution(relu1, **convargs)

    return to_proto(deconv1)

net = mynet()

print str(net)
