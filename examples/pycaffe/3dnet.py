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

    return to_proto(relu1)

net = mynet()

print str(net)
