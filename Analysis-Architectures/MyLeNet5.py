
###
### From https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37
###

from torch import nn

class LeNet5(nn.Module):

    def printLeNet5Comments(self):

        if self.orig_C5_layer:
            print("Original implementation at LeCunn's paper (feature map size in C5 is 1x1 => 'full' convolution)")
        else:
            print("Implementation at https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37")

    
    def __init__(self, orig_C5_layer = True, num_classes: int = 10):

        self.orig_C5_layer = orig_C5_layer

        super(LeNet5, self).__init__()
        self.features = nn.Sequential(  # input: 32x32
            # Block 1
            # C1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),  # 32*32 -> 28*28
            nn.Tanh(),
            # S2
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28*28 -> 14*14
            
            # Block 2
            # C3
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 14*14 -> 10*10
            nn.Tanh(),
            # S4
            nn.AvgPool2d(kernel_size=2, stride=2),  # 10*10 -> 5*5
            
        )
        if self.orig_C5_layer:
            ### Original implementation at LeCunn's paper (feature map size in C5 is 1x1 => "full" convolution)
            self.classifier = nn.Sequential(
                # C5
                nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),  # 5*5 -> 1 (120 channels)
                nn.Flatten(),
                nn.Tanh(),
                # F6
                nn.Linear(in_features=120, out_features=84),  # 120 -> 84
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=num_classes),  # 84 -> num_classes (= 10 in the original task)
            )
        else:
            ### This is the implementation at https://medium.com/@deepeshdeepakdd2/lenet-5-implementation-on-mnist-in-pytorch-c6f2ee306e37
            ###   (equivalent to the original one)
            self.classifier = nn.Sequential(
                # C5
                nn.Flatten(),
                nn.Linear(in_features=16*5*5, out_features=120),
                nn.Tanh(),
                # F6
                nn.Linear(in_features=120, out_features=84),
                nn.Tanh(),
                nn.Linear(in_features=84, out_features=num_classes),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

###
#
# from lenet5 import LeNet5
#
# m = LeNet5(10)
#
# print(m)
#
#   LeNet5(
#     (feature): Sequential(
#       (0): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
#       (1): Tanh()
#       (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
#       (3): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
#       (4): Tanh()
#       (5): AvgPool2d(kernel_size=2, stride=2, padding=0)
#     )
#     (classifier): Sequential(
#       (0): Flatten(start_dim=1, end_dim=-1)
#       (1): Linear(in_features=400, out_features=120, bias=True)
#       (2): Tanh()
#       (3): Linear(in_features=120, out_features=84, bias=True)
#       (4): Tanh()
#       (5): Linear(in_features=84, out_features=10, bias=True)
#     )
#   )
#
###
