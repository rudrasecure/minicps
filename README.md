[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/minicps.svg)](https://badge.fury.io/py/minicps)
[![Documentation Status](https://readthedocs.org/projects/minicps/badge/?version=latest)](http://minicps.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status][CS img]][Coverage Status]

[Coverage Status]: https://travis-ci.org/scy-phy/minicps
[CS img]: https://travis-ci.org/scy-phy/minicps.svg
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/hslatman/awesome-industrial-control-system-security)

# MiniCPS

MiniCPS is a framework for Cyber-Physical Systems real-time simulation. It
includes support for physical process and control devices simulation, and
network emulation. It is built on top of
[mininet](https://github.com/mininet/mininet).

MiniCPS is developed by the [SCy-Phy](http://scy-phy.github.io/index.html)
group from SUTD (Singapore University of Design and Technology).

# Anomaly detection using Autoencoders

An Autoencoder is an FNN that consist of three components, namely, the encoder, the bottlekneck, and the decoder. 

![image](https://github.com/rudrasecure/minicps/assets/52862591/a1ee893a-7faf-4030-ae20-234ec753316e)

1. Encoder: a module that compresses the input data into an encoded representation that is typically several orders of magnitude smaller than the input data. The encoder is composed of a set of convolutional blocks followed by pooling modules or simple linear layers that compress the input to the model into a compact section called the "bottleneck" or "code".

2. Code: As you probably have already understood, the most important part of the neural network, and ironically the smallest one, is the code. The code exists to restrict the flow of information to the decoder from the encoder, thus, allowing only the most vital information to pass through. The code helps us form a knowledge representation of the input.

The code, as a compressed representation of the input, prevents the neural network from memorizing the input and overfitting the data. The smaller the code, the lower the risk of overfitting. This layer is generally implemented as a linear layer or as a tensor if we use convolutions.

3. Decoder: The decoder component of the network acts as an interpreter for the code.

![image](https://github.com/rudrasecure/minicps/assets/52862591/c1021a2f-7da8-4fb8-95ac-010759a1a8b6)


The decoder helps the network to “decompress” the knowledge representations and reconstruct the data back from its encoded form. The output is then compared with the ground truth.

While there are various forms of autoencoders such as VAEs, Contractive Autoencoders, DeepAutoencoders, etc, for the purpose of simplicity, we have used simple undercomplete Autoencoders.

### Training

The code used to generate the Autoencoder is:

```python
from numpy.linalg import norm
from numpy import dot
from keras import layers
import keras
import tensorflow as tf
import pandas as pd
import numpy as np

# read the log data 
data = pd.read_csv("data.csv")
data = data.drop(labels=["Time"], axis=1) # discard time column, since detection does not depend on time
print(data)

# create a sequential model as shown, having 6 hidden layes
# the input and the output need to have 6 inputs since there are 6 fields that we are considering 
autoencoder = keras.Sequential(
    [
        layers.Dense(6, activation="relu", name="layer1"),
        layers.Dense(5, activation="relu", name="layer2"),
        layers.Dense(4, activation="relu", name="layer3"),
        layers.Dense(3, activation="relu", name="layer4"),
        layers.Dense(2, activation="relu", name="layer5"),
        layers.Dense(4, activation="relu", name="layer6"),
        layers.Dense(units=5, activation="relu", name="layer7"),
        layers.Dense(units=6, activation="sigmoid", name="layer8")
    ]
)

ratio = 0.75

# ensure that the training data is not skewed
print(data.describe())

data = np.asarray(data).astype('float32')/255
total_rows = data.shape[0]
train_size = int(total_rows * ratio)

# Split data into test and train
train = data[0:train_size]
test = data[train_size:]

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train the model
autoencoder.fit(train, train,
            epochs=100,
            batch_size=80,
            shuffle=False,
            validation_data=(test, test), verbose=True)

# save the model
autoencoder.save("model.h5")
```


### Detection:




### References

* [Deep learning book](https://www.deeplearningbook.org/)
* [Deep learning](https://deep-learning-study-note.readthedocs.io/en/latest/index.html)

## Reference Research Papers

* [MiniCPS: A toolkit for security research on CPS networks (ACM CPS-SPC15)](https://arxiv.org/pdf/1507.04860)
* [Towards high-interaction virtual ICS honeypots-in-a-box (ACM CPS-SPC16)](https://dl.acm.org/citation.cfm?id=2994493)
* [Gamifying ICS Security Training and Research: Design, Implementation, and Results of S3 (ACM CPS-SPC17)](https://dl.acm.org/citation.cfm?id=3140253)


## Docs

See [readthedocs](http://minicps.readthedocs.io/en/latest/?badge=latest)

## Contributing

See [Contributing](https://minicps.readthedocs.io/en/latest/contributing.html)
