import torch.nn as nn

from config import NUM_CLASSES


class DPCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1,1)
            )
        )

        self.classifier = nn.Linear(
            128,
            NUM_CLASSES
        )

    def forward(
        self,
        x
    ):

        x = self.features(x)

        x = x.view(
            x.size(0),
            -1
        )

        return self.classifier(x)