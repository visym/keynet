# keynet
Key-Nets: Optical Transformation Convolutional Networks for Privacy Preserving Vision Sensors

# Dependencies
python-3.*

Required 

```python
pip3 install torch torchvision numpy scipy vipy scikit-learn joblib tqdm
```

Optional 

```python
pip3 install cupy ipython
```

# Quickstart
```python
net = LeNet_AvgPool()
(sensor, knet) = PermutationKeynet(inshape=(1,28,28), net=net(), do_output_encryption=False)
y = knet.forward(sensor.load('myimage.jpg').encrypt().tensor())
```

