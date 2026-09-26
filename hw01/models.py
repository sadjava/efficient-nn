from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 64, 5, 1, 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        return self.net(x)
