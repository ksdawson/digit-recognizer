# Digit Recognizer
This project implements a solution to the digit recognizer competition on [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/).

## Background

Digit classification is a classic computer vision problem. The task is given an image of a handwritten digit (0-9) classify the image as the digit it shows. The standard approach to computer vision problems like this is convolutional neural networks (CNN). Famously, Yann LeCun developed one of the early CNNs precisely for this problem (see his [demo](https://youtu.be/FwFduRA_L6Q?si=gPfS18NTugDUCg09)). This project implements several increasingly advanced models with better performance on the competition's leaderboard.

## Model #1: Basic CNN

Traditional neural networks treat inputs elements independently, which performs poorly on grid-like data such as images where important context is needed from neighbor elements. CNNs address this with the convolutional layer where a filter/kernel (a matrix of weights) is "slid" across the image to create the inputs for the next layer. The filter helps take into account spatial context from adjacent pixels. The filter is usually very small relative to the image (e.g. 3x3, 5x5, or 7x7), which makes the CNN more efficient as it reuses the weights in the filter for the entire image, rather than using one image-wide filter. The small filter helps learn small features, such as lines or edges. Deeper layers combine results from early layers to learn more complex features such as shapes. CNNs are also invariant to location, transforms, etc so they can learn that a rotated cat in the top left corner is still a cat. You can read more about CNNs [here](https://www.geeksforgeeks.org/machine-learning/introduction-convolution-neural-network/).

The traditional architecture of CNNs is the convolutional layer (described above) followed by a max pooling layer then a fully connected layer before outputting the result, with each layer connected by activation functions. Pooling reduces the size of the outputs from the convolutional layers to speed up computation, reduce memory, and prevent overfitting. In the first model we use this traditional structure with a convolution layer connected to a pool layer, followed by another conv-pool, and finally three linear layers in the fully connected layer. Each layer is connected by a ReLU activation function. ReLU is max(0, x) and helps introduce non-linearity in the network. The convolutional layer uses 5x5 filters. The full architecture is shown below:

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

The network uses cross entropy loss and the stochastic gradient descent (SGD) optimizer.

This version scores **97.928%** on the leaderboard test set ([source](https://www.kaggle.com/code/kamerondawson/digitrecognizer?scriptVersionId=297794776)).

## Model #2: Optimizer and Learning Rate

In this model we switch from the SGD to Adam optimizer. We also add a learning rate scheduler, specifically the ReduceLROnPlateau. Learning rate is a hyperparameter that controls how much the model's weights are updated in response to the estimated error each time they are adjusted. It's the size of the steps taken toward a minimum of the loss function during gradient descent. A small rate makes training slow but stable, whereas large rates make training fast but can be unstable or diverge. The scheduler reduces the learning rate as the validation loss plateaus so it settles into the loss function minimum.

This version scores **98.757%** on the leaderboard test set ([source](https://www.kaggle.com/code/kamerondawson/digitrecognizer?scriptVersionId=297797143)).

## Model #3: Model Architecture

This model improves upon the basic architecture by adding more layers to make the network deeper and able to learn more complex features. It also adds batch normalization and dropout layers and reduces the kernel size from 5x5 to 3x3. Batch normalization stabilizes and accelerates training by normalizing the inputs to a layer to have a mean of zero and unit variance for every mini-batch. It reduces internal covariate shift, allowing higher learning rates and less sensitivity to initialization. Dropout prevents model overfitting by randomly deactivating a subset of neurons during training. Full architecture is shown below:

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # Block 1: Input 1x28x28 -> Output 32x14x14
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.25)

        # Block 2: Input 32x14x14 -> Output 64x7x7
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout2d(0.25)

        # Fully Connected Classifier
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Flatten and Classify
        x = torch.flatten(x, 1)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.drop3(x)
        x = self.fc2(x)
        return x
```

This version scores **99.207%** on the leaderboard test set ([source](https://www.kaggle.com/code/kamerondawson/digitrecognizer?scriptVersionId=297800078)).