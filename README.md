# Keynet
Key-Nets: Optical Transformation Convolutional Networks for Privacy Preserving Vision Sensors

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



