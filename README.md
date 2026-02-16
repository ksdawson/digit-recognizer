# Digit Recognizer
This project implements a solution to the digit recognizer competition on [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/).

## Background

Digit classification is a classic computer vision problem. The task is given an image of a handwritten digit (0-9) classify the image as the digit it shows. The standard approach to computer vision problems like this is convolutional neural networks (CNN). Famously, Yann LeCun developed one of the early CNNs precisely for this problem (see his [demo](https://youtu.be/FwFduRA_L6Q?si=gPfS18NTugDUCg09)). This project implements several increasingly advanced models with better performance on the competition's leaderboard.

## Models

### Model #1: Basic CNN

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

### Model #2: Optimizer and Learning Rate

In this model we switch from the SGD to Adam optimizer. We also add a learning rate scheduler, specifically the ReduceLROnPlateau. Learning rate is a hyperparameter that controls how much the model's weights are updated in response to the estimated error each time they are adjusted. It's the size of the steps taken toward a minimum of the loss function during gradient descent. A small rate makes training slow but stable, whereas large rates make training fast but can be unstable or diverge. The scheduler reduces the learning rate as the validation loss plateaus so it settles into the loss function minimum.

This version scores **98.757%** on the leaderboard test set ([source](https://www.kaggle.com/code/kamerondawson/digitrecognizer?scriptVersionId=297797143)).

### Model #3: Model Architecture

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

### Model #4: Data Augmentation, Model Ensemble, and Epochs

At this point we have a pretty accurate model, so we have to get creative to get further performance boosts. First, it's likely that the remaining misclassified digits are due to weird distortions and our model is overfitting and unable to generalize to learn these quirky patterns. To help prevent this we augment our train set by rotating, translating, and zooming in on digits to create new distorted samples. We also introduce a homogeneous ensemble of 5 models to reduce errors. The models "soft-vote" on the classification by summing their output logits before applying the max activation function. We also increase the number of epochs from 10 to 30 to give sufficient training time to learn the complex features introduced by the data augmentation.

This version scores **99.571%** on the leaderboard test set ([source](https://www.kaggle.com/code/kamerondawson/digitrecognizer?scriptVersionId=297834615)).

### Model #5: Heterogeneous Model Ensemble

In the previous model we added a model ensemble, but it was homogeneous consisting of 5 of the same model with different initializations. This reduces variance but doesn't introduce any new mode abilities. To improve upon this we switch to using a true ensemble of different models. We use two of the CNN architectures from the previous model, a CNN with a larger 5x5 kernel, a ResNet5, and a spatial transformer CNN. I won't go in-depth on these as I got these models off-the-shelf to show the power of model ensembles.

```python
class BigKernelCNN(nn.Module):
    def __init__(self):
        super(BigKernelCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet5(nn.Module):
    def __init__(self):
        super(ResNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.fc = nn.Linear(64 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class STNCNN(nn.Module):
    def __init__(self):
        super(STNCNN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        x = self.stn(x)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

This version scores **99.610%** on the leaderboard test set ([source](https://www.kaggle.com/code/kamerondawson/digitrecognizer?scriptVersionId=297945682)).

## Conclusion