import torch
import torch.nn as nn

# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        self.model = nn.Sequential(
            
            # Intake
            nn.Conv2d(3,80,3,padding=1),
            nn.ReLU(),
            
            # BatchNorm + Dropout
            nn.BatchNorm2d(80),
            nn.MaxPool2d(2,2),
            # 112x112x80 out
            
            # Narrow + BatchNorm + Dropout
            nn.Conv2d(80,128,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2),
            # 56x56x128 out
            
            # Narrow + BatchNorm + Dropout
            nn.Conv2d(128,256,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2),
            # 28x28x256 out
            
            # Narrow + BatchNorm + Dropout
            nn.Conv2d(256,512,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2,2),
            # 14x14x512 out
            
            # Narrow + BatchNorm + Dropout
            nn.Conv2d(512,1024,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2,2),
            # 7x7x1024 out
            
            # Narrow + BatchNorm + Dropout
            nn.Conv2d(1024,2048,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2048),
            nn.MaxPool2d(7,7),
            nn.Flatten(),        
            # 1x2048 out
            
            # MLP
            nn.Dropout(p=dropout),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(p=dropout),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=dropout),
            nn.Linear(1024,num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2,num_workers = 4)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
