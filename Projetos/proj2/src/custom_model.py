from torch import nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel,self).__init__()
        self.pool_size = 2
        self.kernel_size = 5

        self.network = nn.Sequential(
            
            # Layer 1
            nn.Conv2d(3, 64, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(64, 64, self.kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

        
            # Layer 2
            nn.Conv2d(64, 128, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(128, 128, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(128, 128, self.kernel_size),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),

            # Layer 3
            nn.Conv2d(128, 256, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(256, 256, self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(256, 256, self.kernel_size),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),
            

            nn.AdaptiveAvgPool2d(4),    # With this layer, the image shape is not restricted

            # Layer 4
            nn.Flatten(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(4096, 4)
        )
        
    def forward(self, x): 
        return self.network(x)