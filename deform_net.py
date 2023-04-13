import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Embedder(object):
    # Positional encoding (section 5.1)
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(
                2.0 ** 0.0, 2.0 ** max_freq, steps=N_freqs
            )

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims, i=0):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


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
        self.use_linear = use_linear
        if self.use_linear:
            self.fc = nn.Linear(512, c_dim)

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        x = self.features.maxpool(
            self.features.relu(self.features.bn1(self.features.conv1(x)))
        )  # B, 64, 64, 64
        block1_out = self.features.layer1(x)  # B, 64, 64, 64
        block2_out = self.features.layer2(block1_out)  # B, 128, 32, 32
        block3_out = self.features.layer3(block2_out)  # B, 256, 16, 16
        block4_out = self.features.layer4(block3_out)  # B, 512, 8, 8
        if self.use_linear:
            out = self.features.fc(self.features.avgpool(block4_out))
            out = self.fc(out.reshape(out.shape[0], out.shape[1]))
            return out
        else:
            return x, block1_out, block2_out, block3_out, block4_out


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
        net = net.unsqueeze(0)
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                if len(c.shape) == 2:
                    net_c = self.fc_c[n](c)  # b, n_feat -> b, 512
                    if batchwise:
                        net_c = net_c.unsqueeze(1)  # b, 1, 512
                    net = net + net_c  #
                elif len(c.shape) == 3:
                    net_c = self.fc_c[n](
                        c.reshape(-1, c.shape[-1])
                    )  # b, n_feat -> b, 512
                    net_c = net_c.reshape(c.shape[0], c.shape[1], -1)
                    net = net + net_c
            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))

        out_bxpxc = out
        return out_bxpxc


"""
class DeformNet(nn.Module):
    def __init__(
        self,
        tplt_vtx,
        use_depth=False,
        use_normals=False,
        c_dim=256,
        predict_rgb=False,
        multires=10,
    ):
        super(DeformNet, self).__init__()

        self.c_dim = c_dim
        self.rgb_encoder = Resnet18(
            c_dim=self.c_dim, normalize=True, use_linear=True, pretrained=True
        )
        self.decoder_input_dim = self.c_dim

        if use_depth:
            self.embedder, self.depth_out_dim = get_embedder(multires, input_dims=1)
            self.depth_encoder = Resnet18(
                c_dim=self.c_dim,
                # normalize=True,
                # pretrained=True,
                normalize=False,
                pretrained=True,
                use_linear=True,
            )
            self.depth_encoder.features.conv1 = nn.Conv2d(
                self.depth_out_dim,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
            self.decoder_input_dim += self.c_dim

        else:
            self.depth_encoder = None
            self.embedder = None
        if use_normals:
            self.normals_encoder = Resnet18(
                c_dim=self.c_dim,
                normalize=True,
                use_linear=True,
                pretrained=True,
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

    # def forward(self, rgb_images, depth_images, normals_images):
    def forward(self, rgb_images, disparity, normals_images):

        # encode inputs
        c_bxc = self.rgb_encoder(rgb_images)
        if self.normals_encoder is not None:
            normals_feat = self.normals_encoder(normals_images)
            c_bxc = torch.cat([c_bxc, normals_feat], dim=1)
        if self.depth_encoder is not None:
            bs, h, w = disparity.shape
            disparity_feat = self.embedder(disparity.reshape(bs, -1))
            disparity_feat = disparity_feat.reshape(bs, -1, h, w)
            disparity_feat = self.depth_encoder(disparity_feat)
            c_bxc = torch.cat([c_bxc, disparity_feat], dim=1)
            # depth_feat = self.depth_encoder(
            #    disparity.unsqueeze(1).repeat([1, 3, 1, 1])
            # )
            # c_bxc = torch.cat([c_bxc, depth_feat], dim=1)

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
"""


class DeformNet(nn.Module):
    def __init__(
        self,
        tplt_vtx,
        use_depth=False,
        use_normals=False,
        c_dim=256,
        multires=10,
    ):
        super(DeformNet, self).__init__()

        self.c_dim = c_dim
        self.tplt_vtx = nn.Parameter(tplt_vtx, requires_grad=False)

        self.rgb_encoder = Resnet18(
            c_dim=self.c_dim, normalize=True, use_linear=True, pretrained=True
        )
        self.decoder_input_dim = self.c_dim

        if use_depth:
            self.depth_encoder = Resnet18(
                c_dim=self.c_dim,
                # normalize=True,
                # pretrained=True,
                normalize=False,
                pretrained=True,
                use_linear=True,
            )
            self.decoder_input_dim += self.c_dim

        else:
            self.depth_encoder = None
            # self.embedder = None
        if use_normals:
            self.normals_encoder = Resnet18(
                c_dim=self.c_dim,
                normalize=True,
                use_linear=True,
                pretrained=True,
            )
            self.decoder_input_dim += self.c_dim
        else:
            self.normals_encoder = None

        self.embedder, self.embed_out_dim = get_embedder(multires, input_dims=2)
        self.decoder_input_dim += self.embed_out_dim

        out_dim = 3
        decoder = Decoder(
            dim=3,
            c_dim=self.decoder_input_dim,
            leaky=False,  # True,
            out_dim=out_dim,
            res0=True,
            res0ini=torch.ones,
        )
        self.decoder = decoder

    def forward(self, rgb_images, disparity, normals_images, verts_uv):

        # encode inputs
        global_feat = self.rgb_encoder(rgb_images)
        if self.normals_encoder is not None:
            normals_feat = self.normals_encoder(normals_images)
            global_feat = torch.cat([global_feat, normals_feat], dim=1)
        if self.depth_encoder is not None:
            depth_feat = self.depth_encoder(
                disparity.unsqueeze(1).repeat([1, 3, 1, 1])
            )
            global_feat = torch.cat([global_feat, depth_feat], dim=1)
        global_feat = global_feat.unsqueeze(1).repeat(
            [1, verts_uv.shape[0], 1]
        )  # B, N, 256 * 3

        verts_uv = self.embedder(verts_uv)  # N, 42
        verts_uv = verts_uv.unsqueeze(0).repeat([global_feat.shape[0], 1, 1])
        enc_feat = torch.cat([global_feat, verts_uv], dim=-1)

        # decode prediction
        delta_vtx = self.decoder(self.tplt_vtx, c=enc_feat)
        p = self.tplt_vtx + delta_vtx

        return p, delta_vtx, None
