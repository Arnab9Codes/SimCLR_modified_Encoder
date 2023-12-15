import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError
import timm  # If not used, this import can be removed


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
            "squeezenet": models.squeezenet1_0(pretrained=False)
        }
        print('outdim: ',out_dim)

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self._get_dim_mlp(base_model)

        if 'squeezenet' in base_model:

            inC=self.backbone.classifier[1].in_channels
            additional_fc_layer = nn.Sequential(nn.Linear(1000, 512), nn.ReLU(), nn.Linear(512, out_dim))

            # Access the existing classifier and add the additional FC layer
            #self.backbone.add_module('fc', additional_fc_layer)
            #self.backbone.classifier[1] = nn.Conv2d(512, out_dim, kernel_size=1, stride=1)
            #self.backbone.num_classes = out_dim

            self.backbone=nn.Sequential(self.backbone, additional_fc_layer)
            #print(self.backbone.1)

        else:
            if 'resnet' in base_model:
                dim_mlp = self.backbone.fc.in_features
                print('dim_nlp:', dim_mlp)

            # Modify the projection head based on the backbone type
            if 'resnet' in base_model:
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
            else:
                raise InvalidBackboneError(
                    "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet50, squeezenet"
                )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet50, squeezenet")
        else:
            return model

    def _get_dim_mlp(self, model_name):
        model = self.resnet_dict[model_name]
        if 'resnet' in model_name:
            return model.fc.in_features
        elif 'squeezenet' in model_name:
            return 512  # SqueezeNet's number of features before the classifier
        else:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet50, squeezenet")

    def forward(self, x):
        return self.backbone(x)
