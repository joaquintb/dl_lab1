
###
### Constructed from scratch
###   See also:
###     https://stackoverflow.com/questions/40060949/how-to-calculate-the-number-of-parameters-of-alexnet
###

from torch import nn

class alexNetOriginal(nn.Module):

    def __init__(self, num_classes: int = 1000):

        super(alexNetOriginal, self).__init__()

        self.features = nn.Sequential(  # input: 224x224x3
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=1),  # 224x224 -> 55x55x96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 55x55 -> 27x27x96
            # Block 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),  # 27x77 -> 27x27x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27 -> 13x13x256
            # Blocks 3,4,5
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),  # 13x13 -> 13x13x384
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),  # 13x13 -> 13x13x384
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),  # 13x13 -> 13x13x256
            nn.ReLU(),
            # Max-Pooling
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 13x13 -> 6x6x256    (without padding=1 dimensions do not match)
        )
        self.classifier = nn.Sequential(
            # Flatten
            nn.Flatten(),
            # Fully Connected
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(),
            # Fully Connected
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            # Fully Connected
            nn.Linear(in_features=4096, out_features=1000),
        )

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = self.classifier(x)
        return x

###
