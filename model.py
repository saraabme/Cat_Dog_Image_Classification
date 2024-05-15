import torch

class ConvNet(torch.nn.Module):
    # Nested Block class within ConvNet to define a residual block with downsampling if needed
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1):
            super().__init__()
            # Define a sequential container of layers:
            # 1. Convolutional layer
            # 2. Batch normalization layer
            # 3. ReLU activation layer
            # Repeat the Conv-BN-ReLU pattern for a second time
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU(),
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                torch.nn.ReLU()
            )
            # Optional downsampling to match the dimensions of input and output
            self.downsample = None
            if stride != 1 or n_input != n_output:
                self.downsample = torch.nn.Sequential(
                    torch.nn.Conv2d(n_input, n_output, 1, stride=stride),
                    torch.nn.BatchNorm2d(n_output)
                )

        def forward(self, x):
            # Preserve input for skip connection
            identity = x
            # Apply downsampling if defined
            if self.downsample is not None:
                identity = self.downsample(x)
            # Return the output of the block by adding the original input to the output of the net
            return self.net(x) + identity

    def __init__(self, layers=[32, 64, 128], n_input_channels=3):
        super().__init__()
        # Initial layers of the network
        L = [
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        ]
        # Create a series of blocks with increasing number of filters and downsampling
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
            c = l
        # Sequentially stack all layers
        self.network = torch.nn.Sequential(*L)
        # Final classification layer
        self.classifier = torch.nn.Linear(c, 1)

    def forward(self, x):
        # Compute the features through the network
        z = self.network(x)
        # Global average pooling to reduce the spatial dimensions to one scalar per feature map
        z = z.mean(dim=[2, 3])
        # Classify and return the result
        return self.classifier(z)[:, 0]
