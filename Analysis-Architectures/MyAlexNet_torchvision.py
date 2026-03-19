
###
### From ~/tools/anaconda/envs/.../lib/python3.6/site-packages/torchvision/models
###

from torchvision import models, datasets

def printAlexNetComments():

    print("\
      \n\
      *** THIS IS NOT THE SAME ARCHITECTURE AS DESCRIBED IN THE ORIGINAL PAPER ***\n\
      \n\
      AlexNet(\n\
       (features): Sequential(\n\
              *** In the original paper: 96 kernels\n\
         (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))\n\
              *** In the original paper: ReLU followed by Local Response Normalization\n\
         (1): ReLU(inplace=True)\n\
         (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n\
              *** In the original paper: 256 kernels\n\
         (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))\n\
              *** In the original paper: ReLU followed by Local Response Normalization\n\
         (4): ReLU(inplace=True)\n\
         (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n\
         (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n\
         (7): ReLU(inplace=True)\n\
              *** In the original paper: 384 kernels\n\
         (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n\
         (9): ReLU(inplace=True)\n\
         (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n\
         (11): ReLU(inplace=True)\n\
         (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)\n\
       )\n\
              *** In the original paper: This layer is not present\n\
              *** A 2-D adaptive average pooling layer performs downsampling to give you\n\
              *** the desired output size by dividing the input into rectangular pooling\n\
              *** regions, then computing the average of each region.\n\
       (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))\n\
      \n\
       (classifier): Sequential(\n\
         (0): Dropout(p=0.5, inplace=False)\n\
         (1): Linear(in_features=9216, out_features=4096, bias=True)\n\
              *** 9216 = 6*6*256 (6*6 comes from the AdaptiveAvgPool2d, 256 is the number of channels)\n\
         (2): ReLU(inplace=True)\n\
         (3): Dropout(p=0.5, inplace=False)\n\
         (4): Linear(in_features=4096, out_features=4096, bias=True)\n\
         (5): ReLU(inplace=True)\n\
         (6): Linear(in_features=4096, out_features=10, bias=True)\n\
       )\n\
      )\n\
    ")

### The rest is as usual
#
# m = models.alexnet(num_classes=1000)
#
# print(m)
#
#   AlexNet(
#     (features): Sequential(
#       (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))            ### In the original paper: 96 kernels
#       (1): ReLU(inplace=True)                                                            ### In the original paper: ReLU followed by Local Response Normalization
#       (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))            ### In the original paper: 256 kernels
#       (4): ReLU(inplace=True)                                                            ### In the original paper: ReLU followed by Local Response Normalization
#       (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#       (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (7): ReLU(inplace=True)
#       (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))           ### In the original paper: 384 kernels
#       (9): ReLU(inplace=True)
#       (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (11): ReLU(inplace=True)
#       (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
#     )
#     (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))                                      ### In the original paper: This layer is not present
#                                                                                           ### A 2-D adaptive average pooling layer performs downsampling to give you
#                                                                                           ### the desired output size by dividing the input into rectangular pooling
#                                                                                           ### regions, then computing the average of each region.
#
#     (classifier): Sequential(
#       (0): Dropout(p=0.5, inplace=False)
#       (1): Linear(in_features=9216, out_features=4096, bias=True)
#              *** 9216 = 6*6*256 (6*6 comes from the AdaptiveAvgPool2d, 256 is the number of channels)
#       (2): ReLU(inplace=True)
#       (3): Dropout(p=0.5, inplace=False)
#       (4): Linear(in_features=4096, out_features=4096, bias=True)
#       (5): ReLU(inplace=True)
#       (6): Linear(in_features=4096, out_features=10, bias=True)
#     )
# )
#
###

##################################################################################################################3

### We can remove this...

# import torch
# import torch.nn as nn
# from typing import Any
#
# #__all__ = ['AlexNet', 'alexnet']
#
#
# class AlexNet(nn.Module):
#
#     ###
#     ### *THIS IS NOT THE SAME ARCHITECTURE AS DESCRIBED IN THE ORIGINAL PAPER*
#     ###
#     def __init__(self, num_classes: int = 1000) -> None:
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),   ### In the original paper: 96 kernels
#             nn.ReLU(inplace=True),                                   ### In the original paper: ReLU followed by Local Response Normalization
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),            ### In the original paper: 256 kernels
#             nn.ReLU(inplace=True),                                   ### In the original paper: ReLU followed by Local Response Normalization
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),           ### In the original paper: 384 kernels
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                  ### In the original paper: This layer is not present
#                                                                      ### A 2-D adaptive average pooling layer performs downsampling to give you
#                                                                      ### the desired output size by dividing the input into rectangular pooling
#                                                                      ### regions, then computing the average of each region.
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
#

#
# def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     The required minimum input size of the model is 63x63.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         progress (bool): If True, displays a progress bar of the download to stderr
#     """
#     model = AlexNet(**kwargs)
#     if pretrained:
#         from .._internally_replaced_utils import load_state_dict_from_url
#         model_urls = {
#             'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
#         }
#         state_dict = load_state_dict_from_url(model_urls['alexnet'],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model

###
