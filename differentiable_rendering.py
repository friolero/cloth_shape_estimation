import glob
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.cameras import (FoVPerspectiveCameras,
                                        look_at_view_transform)
from pytorch3d.renderer.lighting import (AmbientLights, DirectionalLights,
                                         PointLights)
from pytorch3d.renderer.mesh.rasterizer import (MeshRasterizer,
                                                RasterizationSettings)
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import (SoftPhongShader,
                                            SoftSilhouetteShader)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def plot_image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    mode: str = "rgb",
    display: bool = False,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(
        rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9)
    )
    if (rows == 1) and (cols == 1):
        axarr = np.array([axarr], dtype=np.object)
    bleed = 0
    fig.subplots_adjust(
        left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed)
    )

    for ax, im in zip(axarr.ravel(), images):
        if mode == "rgb":
            # only render RGB channels
            ax.imshow(im[..., :3])
        elif mode == "depth":
            norm_im = im - im[..., 0].min() / (
                im[..., 0].max() - im[..., 0].min()
            )
            ax.imshow(im[..., 0])
            ax.imshow(norm_im[..., 0])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()
    if display:
        ax.set_title(mode)
        plt.show()


class CameraInterface:
    def __init__(
        self,
        device,
        image_size,
        cam_dist,
        elevation,
        azimuth,
        lights,
        sigma=1e-4 / 3,
        faces_per_pixel=50,
        mode=["rgb", "depth", "silhouette"],
        bg_color=(0.0, 0.0, 0.0),
    ):

        self.device = device

        self.lights = lights
        self.image_size = image_size
        self.sigma = sigma
        self.blur_radius = np.log(1.0 / 1e-4 - 1.0) * self.sigma
        self.faces_per_pixel = faces_per_pixel
        self.blend_params = BlendParams()
        self.blend_params = self.blend_params._replace(
            background_color=bg_color
        )
        self.update_camera(cam_dist, elevation, azimuth)

        self.hard_raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        sigma = 1e-4 / 3
        self.soft_raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel,
        )

        self.renderers = []
        self.mode = mode
        for mode in mode:
            assert mode in ["rgb", "depth", "silhouette"], "Unsupported type."
            self.renderers.append(getattr(self, f"init_{mode}_renderer")())

    def update_camera(self, cam_dist, elevation, azimuth):
        self.cam_dist = cam_dist
        self.elevation = elevation
        self.azimuth = azimuth
        assert len(self.elevation) == len(self.azimuth), "Mismatched input."
        R, T = look_at_view_transform(
            dist=self.cam_dist, elev=self.elevation, azim=self.azimuth
        )
        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        self.num_cameras = len(self.cameras)
        self.param_camera = self.cameras[np.random.randint(0, self.num_cameras)]

    def init_rgb_renderer(self, camera=None):
        if camera is None:
            camera = self.cameras
        rgb_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=self.hard_raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=camera,
                blend_params=self.blend_params,
            ),
        )
        return rgb_renderer

    def init_depth_renderer(self, camera=None):
        if camera is None:
            camera = self.cameras
        """
        # This is not working as intended. Do not use but use z-buffer directly
        depth_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=self.soft_raster_settings
            ),
            shader=SoftDepthShader(
                device=self.device,
                cameras=camera,
            ),
        )
        """
        depth_rasterizer = MeshRasterizer(
            cameras=camera, raster_settings=self.soft_raster_settings
        )
        return depth_rasterizer

    def init_silhouette_renderer(self, camera=None):
        if camera is None:
            camera = self.cameras
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=self.soft_raster_settings
            ),
            shader=SoftSilhouetteShader(
                blend_params=self.blend_params,
            ),
        )
        return silhouette_renderer

    def update_lights(self, lights):
        self.lights = lights

    def render(self, meshes, cameras=None, lights=None, vis=False):
        if cameras is None:
            cameras = self.cameras
        if lights is None:
            lights = self.lights
        assert len(meshes) == len(cameras), "Input sizes mismatch."
        results = {}
        for mode, renderer in zip(self.mode, self.renderers):
            results[mode] = renderer(meshes, cameras=cameras, lights=lights)
            if mode == "depth":
                results[mode] = results[mode].zbuf[:, :, :, :1]
            if vis:
                for nc in range(math.ceil(math.sqrt(self.num_cameras)), 0, -1):
                    nr = int(self.num_cameras / nc)
                    if (nc * nr) == self.num_cameras:
                        break
                plot_image_grid(
                    results[mode].detach().cpu().numpy(),
                    rows=nr,
                    cols=nc,
                    mode=mode,
                    display=True,
                )
        return results


def init_lighting(mode, device, **kwargs):
    assert mode in ["point", "directional", "ambient"], "Wrong lighting mode"
    if mode == "point":
        lights = PointLights(
            device=device, location=[[0.0, 0.0, -3.0]], **kwargs
        )
    elif mode == "directional":
        lights = DirectionalLights(device=device)
    elif mode == "ambient":
        lights = AmbientLights(
            device=device, ambient_color=np.random.uniform(size=3), **kwargs
        )
    return lights


if __name__ == "__main__":

    base_dir = "/home/zyuwei/Projects/cloth_shape_estimation/data/"
    cano_obj_fn = f"{base_dir}/textured_flat_cloth.obj"
    rand_obj_files = glob.glob(f"{base_dir}/perturb*.obj")

    obj_fn = np.random.choice(rand_obj_files, size=1, replace=False)
    mesh = load_objs_as_meshes(obj_fn, device=device)

    num_views = 4
    cam_dist = 8
    image_size = 512
    elevation = torch.linspace(0, 360, num_views)
    azimuth = torch.linspace(-180, 180, num_views)
    lights = init_lighting(mode="point", device=device)
    camera = CameraInterface(
        device, image_size, cam_dist, elevation, azimuth, lights
    )

    meshes = mesh.extend(num_views)
    outcome = camera.render(meshes, vis=True)
    import ipdb

    ipdb.set_trace()
