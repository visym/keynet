import keynet.sensor
import keynet.model


def IdentityKeynet(inshape, net):
    sensor = keynet.sensor.IdentityKeysensor(inshape)
    knet = keynet.model.IdentityKeynet(net, inshape, inkey=sensor.key())
    return (sensor, knet)


def PermutationKeynet(inshape, net, do_output_encryption=False):
    sensor = keynet.sensor.PermutationKeysensor(inshape)
    knet = keynet.model.PermutationKeynet(net, inshape, inkey=sensor.key(), do_output_encryption=do_output_encryption)
    return (sensor, knet)


def StochasticKeynet(inshape, net, alpha, beta=0, do_output_encryption=False):
    sensor = keynet.sensor.StochasticKeysensor(inshape, alpha, beta)    
    knet = keynet.model.StochasticKeynet(net, inshape, inkey=sensor.key(), do_output_encryption=do_output_encryption, alpha=alpha, beta=beta)
    return (sensor, knet)


def PermutationTiledKeynet(inshape, net, tilesize, do_output_encryption=False):
    sensor = keynet.sensor.PermutationTiledKeysensor(inshape, tilesize)
    knet = keynet.model.PermutationTiledKeynet(net, inshape, inkey=sensor.key(), tilesize=tilesize, do_output_encryption=do_output_encryption)
    return (sensor, knet)


def IdentityTiledKeynet(inshape, net, tilesize, do_output_encryption=False):
    sensor = keynet.sensor.IdentityTiledKeysensor(inshape, tilesize)
    knet = keynet.model.IdentityTiledKeynet(net, inshape, inkey=sensor.key(), tilesize=tilesize, do_output_encryption=do_output_encryption)
    return (sensor, knet)


def StochasticTiledKeynet(inshape, net, tilesize, alpha, beta=0, do_output_encryption=False):
    sensor = keynet.sensor.StochasticTiledKeysensor(inshape, tilesize, alpha, beta)    
    knet = keynet.model.StochasticTiledKeynet(net, inshape, tilesize=tilesize, inkey=sensor.key(), do_output_encryption=do_output_encryption, alpha=alpha, beta=beta)
    return (sensor, knet)



    
    
    
