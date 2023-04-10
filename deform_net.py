import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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


class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        res0 (bool): use learnable resnet or not
        res0ini (callable): initilization methods for learnable resnet
    """

    def __init__(
        self,
        size_in,
        size_out=None,
        size_h=None,
        res0=False,
        res0ini=torch.zeros,
    ):
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

        if res0:
            alpha = res0ini(1)
            alpha.requires_grad = True
            alpha = nn.Parameter(alpha)
        else:
            alpha = 1
        self.alpha = alpha

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + self.alpha * dx


class Decoder(nn.Module):
    """Decoder class.

    As discussed in the paper, we implement the OccupancyNetwork
    f and TextureField t in a single network. It consists of 5
    fully-connected ResNet blocks with ReLU activation.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
        out_dim (int): output dimension (e.g. 1 for only
            occupancy prediction or 4 for occupancy and
            RGB prediction)
        res0 (bool): use learnable resnet or not
        res0ini (callable): initialization methods for learnable resnet
    """

    def __init__(
        self,
        dim=3,
        c_dim=128,
        hidden_size=512,
        leaky=False,
        n_blocks=5,
        out_dim=4,
        res0=False,
        res0ini=torch.zeros,
    ):
        super(Decoder, self).__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for i in range(n_blocks)]
            )

        self.blocks = nn.ModuleList(
            [
                ResnetBlockFC(hidden_size, res0=res0, res0ini=res0ini)
                for i in range(n_blocks)
            ]
        )

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(
        self,
        p,
        c=None,
        batchwise=True,
        only_occupancy=False,
        only_texture=True,
        **kwargs
    ):

        assert (len(p.shape) == 3) or (len(p.shape) == 2)

        net = self.fc_p(p)
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c)
                if batchwise:
                    net_c = net_c.unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))

        out_bxpxc = out
        return out_bxpxc


class DeformNet(nn.Module):
    """Deformation prediction class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        tplt_vtx (torch.FloatTensor): template vertices (num_vertices, 3)
    """

    def __init__(
        self,
        tplt_vtx,
        use_depth=False,
        use_normals=False,
        c_dim=256,
        predict_rgb=False,
    ):
        super(DeformNet, self).__init__()

        self.c_dim = c_dim
        self.rgb_encoder = Resnet18(
            c_dim=self.c_dim, normalize=True, use_linear=True, pretrained=True
        )
        self.decoder_input_dim = self.c_dim

        if use_depth:
            self.depth_encoder = Resnet18(
                c_dim=self.c_dim,
                normalize=False,
                use_linear=True,
                pretrained=False,
            )
            self.decoder_input_dim += self.c_dim
        else:
            self.depth_encoder = None
        if use_normals:
            self.normals_encoder = Resnet18(
                c_dim=self.c_dim,
                normalize=False,
                use_linear=True,
                pretrained=False,
            )
            self.decoder_input_dim += self.c_dim
        else:
            self.normals_encoder = None

        out_dim = 3
        self.predict_rgb = predict_rgb
        if predict_rgb:
            out_dim += 3
        decoder = Decoder(
            dim=3,
            c_dim=self.decoder_input_dim,
            leaky=True,
            out_dim=out_dim,
            res0=True,
            res0ini=torch.ones,
        )
        self.decoder = decoder

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
        pred = self.decoder(self.tplt_vtx, c=c_bxc)
        delta_vtx = pred[:, :, :3]  # offset
        # p = scale.unsqueeze(1) * self.tplt_vtx + self.residual_coef * delta_vtx
        p = self.tplt_vtx + delta_vtx

        # optional color prediction
        if self.predict_rgb:
            rgb = F.sigmoid(pred[:, :, 3:6])  # color as texture
            return p, delta_vtx, rgb
        else:
            return p, delta_vtx, None
