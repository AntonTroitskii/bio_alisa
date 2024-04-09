import torch
from torch import nn
from torchvision import transforms
from torchvision.models import (resnet18, resnet34, resnet50, resnet101,
                                resnet152)

torch_models = {'resnet18': resnet18, 'resnet34': resnet34,
                'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152}

resnet_cl_input_shape = {'resnet18': 512, 'resnet34': resnet34,
                         'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152}


def get_fine_tune_model(params, model):

    model_name = params['train']['nn']
    num_classes = params['base']['num_classes']

    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model.parameters():
        param.requires_grad = False

    if 'resnet' in model_name:
        model.fc = torch.nn.Linear(in_features=model.fc.in_features,
                                   out_features=num_classes)
    return model


def get_torch_model(params, device):

    model_name = params['train']['nn']
    model_weight = params['train']['weights']

    if model_name in torch_models:
        model = torch_models[model_name](
            weights=model_weight).to(device=device)
        if params['train']['fine_tune']:
            model = get_fine_tune_model(params=params, model=model)
    else:
        print("There is no such model")

        # Recreate the classifier layer and seed it to the target devic

    return model


class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),  # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x


def create_tinyvgg():
    tinyVGG = TinyVGG()

    return tinyVGG


def get_simple_image_transform():
    return transforms.Compose(
        [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    )
