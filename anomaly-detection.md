
# Anomaly detection using Autoencoders

An Autoencoder is an FNN that consist of three components, namely, the encoder, the bottlekneck, and the decoder. 

![image](https://github.com/rudrasecure/minicps/assets/52862591/a1ee893a-7faf-4030-ae20-234ec753316e)

_Sure, the above image is of an autoencoder that uses CNNs and is for images. But, it gives a better understanding of the intuition behind Autoencoders, i.e. force the model to retain useful information and discard the rest!_

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

We did our best to sample data as best we could to reduce false positives and biases in the results. But, as always, the models aren't 100% accurate and raise false positives based on the threshold set.

### Detection:

For detection, we simply tail the output of the log file of our system and pass it through the model. We then compare the similarity between the data that was entered by the model and its output.

If the output is very similar to the input, it implies that the model was able to reconstruct the original data to an acceptable degree from the compressed data. 
This in turn means that the model has seen similar data before and hence, this was not an anomaly. Else, flag the input as an anomaly.

```python
import select
import subprocess
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import time

# convert list to a Dataframe
def convertToDF(values):
    kv_pair = {"Time": float(values[0]), "MV101": int(values[1]), "P101": float(values[2]), "LIT101": float(
        values[3]), "LIT301": float(values[4]), "FIT101": float(values[5]), "FIT201": float(values[6])}
    dataFrame = pd.DataFrame([kv_pair])
    return dataFrame


# tail last line
def readLastLine(filename):
    line = subprocess.check_output(['tail', '-1', filename])
    return line.decode("utf-8")

def alertGenerator(data):
    print(f"[*] ANOMALY DETECTED. Alert Data:\n{data}\n")

class Detector:

    def __init__(self):
        self.autoencoder = load_model("model.h5")
        self.threshold = 0.64 # threshold can be adjusted based on required sensitivity level

    def detector(self, data):
        data=data.drop(labels=["Time"],axis=1)
        test = np.asarray(data).astype('float32')/255
        result = self.autoencoder.predict(test)
        # calculate cosine similarity. Value > 0.85 is desirable
        a=result[0]
        b=test[0]
        cos_sim = dot(a, b)/(norm(a)*norm(b))
        return cos_sim
    
    # return True to trigger alarm, else do nothing
    def alarm(self, cos_sim):
        if cos_sim > self.threshold:
            return False
        else:
            return True

# initialize the detector outside the loop
detect = Detector()
f = subprocess.Popen(['tail','-n', '+1', '-F', '../examples/swat-s1/logs/data.csv'], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
p = select.poll()
p.register(f.stdout)


while True:
    if p.poll(1):
        line = f.stdout.readline().decode('utf-8')
        lineMod = line.split(",")
        data = convertToDF(lineMod)
        if detect.alarm(detect.detector(data)):
            alertGenerator(data)
```

*How do we determine the similarity between the input and the output?*

![image](https://github.com/rudrasecure/minicps/assets/52862591/512cf847-5d4c-4c1a-8637-98e3a5d165df)

Since both the input and the output are vectors, we calculate the cosine distance between the two.

![image](https://github.com/rudrasecure/minicps/assets/52862591/e76f2af8-4547-4e27-9081-a7f907183c61)

### Could we do better than this?

Yes, we could. VAE, Contractive autoencoders, Sparse Autoencoders, and Denoising autoencoders are options worth exploring. Read more about [autoencoders here](https://www.deeplearningbook.org/contents/autoencoders.html) :)

*Interested in the math behind Autoencoders? Refer to [this article](https://towardsdatascience.com/generating-images-with-autoencoders-77fd3a8dd368) for an in-depth explanation*

### References for Deep Learning

* [Deep learning book](https://www.deeplearningbook.org/)
* [Deep learning](https://deep-learning-study-note.readthedocs.io/en/latest/index.html)
