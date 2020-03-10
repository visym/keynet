import keynet.sensor
import keynet.layer


def IdentityKeynet(inshape, net):
    sensor = keynet.sensor.IdentityKeysensor(inshape)
    knet = keynet.layer.IdentityKeynet(net, inshape, inkey=sensor.key())
    return (sensor, knet)


def PermutationKeynet(inshape, net, do_output_encryption=False):
    sensor = keynet.sensor.PermutationKeysensor(inshape)
    knet = keynet.layer.PermutationKeynet(net, inshape, inkey=sensor.key(), do_output_encryption=do_output_encryption)
    return (sensor, knet)


def StochasticKeynet(inshape, net, alpha, do_output_encryption=False):
    sensor = keynet.sensor.StochasticKeysensor(inshape, alpha=1)    
    knet = keynet.layer.StochasticKeynet(net, inshape, inkey=sensor.key(), do_output_encryption=do_output_encryption, alpha=alpha)
    return (sensor, knet)


def PermutationTiledKeynet(inshape, net, tilesize, do_output_encryption=False):
    sensor = keynet.sensor.PermutationTiledKeysensor(inshape, tilesize)
    knet = keynet.layer.PermutationTiledKeynet(net, inshape, inkey=sensor.key(), tilesize=tilesize, do_output_encryption=do_output_encryption)
    return (sensor, knet)

def IdentityTiledKeynet(inshape, net, tilesize, do_output_encryption=False):
    sensor = keynet.sensor.IdentityTiledKeysensor(inshape, tilesize)
    knet = keynet.layer.IdentityTiledKeynet(net, inshape, inkey=sensor.key(), tilesize=tilesize, do_output_encryption=do_output_encryption)
    return (sensor, knet)



    
    
    
