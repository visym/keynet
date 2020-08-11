# Keynet
Key-Nets: Optical Transformation Convolutional Networks for Privacy Preserving Vision Sensors

![Keynet](./docs/keynet_overview.png)

[[project]](https://visym.github.io/keynet)  [[paper]](http://arxiv.org) 


# Installation
python 3

```python
# Virtualenv installation
pip3 install scipy numpy torch torchvision vipy scikit-learn xxhash numba tqdm
pip3 install -e .  
```

```python
# Virtualenv installation (Linux, multi-threaded scipy/numpy)
pip3 install intel-scipy intel-numpy torch torchvision vipy scikit-learn xxhash numba tqdm
pip3 install -e .  
```

# Quickstart
```python
from keynet.mnist import LeNet_AvgPool
from keynet.system import PermutationKeynet

net = LeNet_AvgPool()
(sensor, knet) = PermutationKeynet(inshape=(1,28,28), net=net, do_output_encryption=False)
y = knet.forward(sensor.load('./demo/owl.jpg').encrypt().tensor())
```

## Citation

**Key-Nets: Optical Transformation Convolutional Networks for Privacy Preserving Vision Sensors**  
Jeffrey Byrne [(Visym Labs)](https://visym.com), Brian Decann [(STR)](https://stresearch.com), Scott Bloom [(STR)](https://stresearch.com)  
British Machine Vision Conference (BMVC) 2020  

> @InProceedings{Byrne2020bmvc,  
>     author       = "J. Byrne and B. Decann and S. Bloom",  
>     title        = "Key-Nets: Optical Transformation Convolutional Networks for Privacy Preserving Vision Sensors",  
>     booktitle    = "British Machine Vision Conference (BMVC)",  
>     year         = "2020"  
> }  
    

## Acknowledgement

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001119C0067. 



