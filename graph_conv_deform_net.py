import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
import torch.utils.model_zoo as model_zoo

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, BatchNorm=None
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, BatchNorm=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, BatchNorm, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm)
        self.layer2 = self._make_layer(
            block, 128, layers[1], BatchNorm, stride=2
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], BatchNorm, stride=1
        )
        self.layer4 = self._make_layer(
            block, 256, layers[3], BatchNorm, stride=1
        )
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, BatchNorm, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, BatchNorm=BatchNorm
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, BatchNorm=nn.BatchNorm2d, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], BatchNorm=BatchNorm, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls["resnet18"])
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if not (k.startswith("layer4") or k.startswith("fc"))
        }
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


def resnet34(pretrained=False, BatchNorm=nn.BatchNorm2d, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], BatchNorm=BatchNorm, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, BatchNorm=nn.BatchNorm2d, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], BatchNorm=BatchNorm, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls["resnet50"])
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if not (k.startswith("layer4") or k.startswith("fc"))
        }
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, BatchNorm=nn.BatchNorm2d, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], BatchNorm=BatchNorm, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls["resnet101"])
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if not (k.startswith("layer4") or k.startswith("fc"))
        }
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, BatchNorm=nn.BatchNorm2d, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], BatchNorm=BatchNorm, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # self.encoder = resnet50(pretrained=True)
        self.encoder = resnet18(pretrained=True)

    def forward(self, img):
        return self.encoder(img)

"""


def normalize_imagenet(x):
    """Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


class Resnet18(nn.Module):
    """ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, normalize=True, use_linear=True, pretrained=True):
        super(Resnet18, self).__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet18(pretrained=pretrained)
        self.features.fc = nn.Identity()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        else:
            self.fc = nn.Identity()
            if c_dim != 512:
                raise ValueError("c_dim must be 512 if use_linear is False")

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        # print(net.shape)
        out = self.fc(net)
        return out


class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907."""

    def __init__(self, in_features, out_features, adjmat, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adjmat = adjmat
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        stdv = 6.0 / math.sqrt(self.weight.size(0) + self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if x.ndimension() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(self.adjmat, support)
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            output = []
            for i in range(x.shape[0]):
                support = torch.matmul(x[i], self.weight)
                # output.append(torch.matmul(self.adjmat, support))
                output.append(spmm(self.adjmat, support))
            output = torch.stack(output, dim=0)
            if self.bias is not None:
                output = output + self.bias
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GraphLinear(nn.Module):
    """
    Generalization of 1x1 convolutions on Graphs
    """

    def __init__(self, in_channels, out_channels):
        super(GraphLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = nn.Parameter(torch.FloatTensor(out_channels, in_channels))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        w_stdv = 1 / (self.in_channels * self.out_channels)
        self.W.data.uniform_(-w_stdv, w_stdv)
        self.b.data.uniform_(-w_stdv, w_stdv)

    def forward(self, x):
        return torch.matmul(self.W[None, :], x) + self.b[None, :, None]


class GraphResBlock(nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """

    def __init__(self, in_channels, out_channels, A):
        super(GraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, out_channels // 2)
        self.conv = GraphConvolution(out_channels // 2, out_channels // 2, A)
        self.lin2 = GraphLinear(out_channels // 2, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        self.pre_norm = nn.GroupNorm(in_channels // 8, in_channels)
        self.norm1 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))
        self.norm2 = nn.GroupNorm((out_channels // 2) // 8, (out_channels // 2))

    def forward(self, x):
        y = F.relu(self.pre_norm(x))
        y = self.lin1(y)

        y = F.relu(self.norm1(y))
        y = self.conv(y.transpose(1, 2)).transpose(1, 2)

        y = F.relu(self.norm2(y))
        y = self.lin2(y)
        if self.in_channels != self.out_channels:
            x = self.skip_conv(x)
        return x + y


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        (sparse,) = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


class GraphCNN(nn.Module):
    def __init__(
        self, A, ref_vertices, infeature=2048, num_layers=5, num_channels=512
    ):
        super(GraphCNN, self).__init__()
        self.A = A
        self.ref_vertices = ref_vertices
        self.infeature = infeature

        layers = [GraphLinear(3 + infeature, 2 * num_channels)]
        layers.append(GraphResBlock(2 * num_channels, num_channels, A))
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, A))
        self.shape = nn.Sequential(
            GraphResBlock(num_channels, 64, A),
            GraphResBlock(64, 32, A),
            nn.GroupNorm(32 // 8, 32),
            nn.ReLU(inplace=True),
            GraphLinear(32, 3),
        )
        self.gc = nn.Sequential(*layers)

    def forward(self, feat):
        """Forward pass
        Inputs:
            x: size = (B, self.infeature)
        Returns:
            Regressed non-parametric displacements: size = (B, 6890, 3)
        """

        batch_size = feat.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)

        x = feat.view(batch_size, self.infeature, 1).expand(
            -1, -1, ref_vertices.shape[-1]
        )
        x = torch.cat([ref_vertices, x], dim=1)
        x = self.gc(x)
        shape = self.shape(x).permute(0, 2, 1)

        return shape


class GraphConvDeformNet(nn.Module):
    """Deformation prediction class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        tplt_vtx (torch.FloatTensor): template vertices (num_vertices, 3)
    """

    def __init__(
        self,
        tplt_vtx,
        adjacency_mtx,
        use_depth=False,
        use_normals=False,
        c_dim=512,
        gcn_layers=5,
        gcn_channels=256,
    ):
        super(GraphConvDeformNet, self).__init__()

        self.c_dim = c_dim

        self.rgb_encoder = Resnet18(
            c_dim=self.c_dim, normalize=True, use_linear=True, pretrained=True
        )
        self.decoder_input_dim = self.c_dim
        if use_depth:
            self.depth_encoder = Resnet18(
                c_dim=self.c_dim,
                normalize=True,
                pretrained=True,
                use_linear=True,
            )
            self.decoder_input_dim += self.c_dim
        else:
            self.depth_encoder = None
        if use_normals:
            self.normals_encoder = Resnet18(
                c_dim=self.c_dim,
                normalize=True,
                pretrained=True,
                use_linear=True,
            )
            self.decoder_input_dim += self.c_dim
        else:
            self.normals_encoder = None

        self.decoder = GraphCNN(
            adjacency_mtx.float(),
            tplt_vtx.t(),  # TO CHECK
            self.decoder_input_dim,
            gcn_layers,
            gcn_channels,
        )

        self.tplt_vtx = nn.Parameter(tplt_vtx, requires_grad=False)

        # learn the delta
        # residual_coef = torch.zeros(1)
        # self.residual_coef = nn.Parameter(residual_coef)

    def forward(self, rgb_images, depth_images, normals_images):

        # encode inputs

        c_bxc = self.rgb_encoder(rgb_images)
        if self.depth_encoder is not None:
            depth_feat = self.depth_encoder(depth_images)
            c_bxc = torch.cat([c_bxc, depth_feat], dim=1)
        if self.normals_encoder is not None:
            normals_feat = self.normals_encoder(normals_images)
            c_bxc = torch.cat([c_bxc, normals_feat], dim=1)

        # decode prediction
        pred = self.decoder(c_bxc)
        delta_vtx = pred[:, :, :3]
        p = self.tplt_vtx + delta_vtx

        return p, delta_vtx, None
