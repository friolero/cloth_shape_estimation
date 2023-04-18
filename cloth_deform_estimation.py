import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

from data_utils import depth2disparity, load_wavefront_file
from differentiable_rendering import CameraInterface, init_lighting


def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, z, -y)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.set_title(title)
    ax.view_init(190, 30)
    # ax.autoscale()
    plt.show()


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    base_dir = "/home/zyuwei/Projects/cloth_shape_estimation/data/"
    cano_obj_fn = f"{base_dir}/textured_flat_cloth.obj"
    rand_obj_files = glob.glob(f"{base_dir}/*/perturb*.obj")
    tgt_obj_fn = rand_obj_files[232]
    tgt_obj_fn = np.random.choice(rand_obj_files)

    tplt_verts, tplt_face_idxs, tplt_tex, tplt_mesh = load_wavefront_file(
        cano_obj_fn, device
    )

    tgt_verts, tgt_face_idxs, tgt_tex, tgt_mesh = load_wavefront_file(
        tgt_obj_fn, device
    )
    # plot_pointcloud(tgt_mesh, "Target mesh")
    # plot_pointcloud(tplt_mesh, "Template mesh")

    image_size = 256
    cam_dist = 8.0
    elevation = [30.0]
    azimuth = [45.0]
    lights = init_lighting(mode="point", device=device)
    camera = CameraInterface(
        device,
        image_size,
        cam_dist,
        elevation,
        azimuth,
        lights,
        mode=["rgb", "depth", "normals"],
    )

    obj_name = "_".join(tgt_obj_fn.split("/")[-1].split(".")[0].split("_")[-2:])
    os.system(f"mkdir -p vis/obj{obj_name}")
    save_prefix = f"vis/obj{obj_name}/groundtruth"
    _ = camera.render(tplt_mesh.extend(len(elevation)), vis=False)
    tgt_render = camera.render(
        tgt_mesh.extend(len(elevation)), vis=False, save_prefix=save_prefix
    )

    zero_dfm_verts = torch.full(
        tplt_mesh.verts_packed().shape,
        0.0,
        device=device,
        requires_grad=True,
    )
    init_verts = {"no_init": zero_dfm_verts}
    if len(sys.argv) > 1:
        from deform_net import DeformNet

        print(f"Loading pre-trained model from {sys.argv[1]}")
        model = DeformNet(
            tplt_verts.cpu(),
            use_depth=True,
            embed_depth=True,
            use_normals=True,
            embed_normals=False,
            embed_uv=True,
            c_dim=256,
        )
        ckpt = torch.load(sys.argv[1], map_location="cpu")["model"]
        model.load_state_dict(ckpt)
        model.eval()

        verts_uv = tplt_mesh.textures.verts_uvs_list()[0].cpu()
        disparity = depth2disparity(
            tgt_render["depth"].cpu().permute(0, 3, 1, 2)
        )
        with torch.no_grad():
            _, dfm_verts = model(
                tgt_render["rgb"].cpu()[..., :3].permute(0, 3, 1, 2),
                disparity,
                tgt_render["normals"].cpu().permute(0, 3, 1, 2),
                verts_uv,
            )
        nn_dfm_verts = dfm_verts[0].to(device).requires_grad_()
        init_verts["deformnet_init"] = nn_dfm_verts
    image_criterion = torch.nn.L1Loss()

    n_iter = 2000
    plot_period = n_iter - 1
    mask_tgt = True
    w_rgb = 1.0
    w_depth = 2.0
    w_normals = 2.0
    w_chamfer = 2.0
    w_edge = 0.0  # not effective
    w_mesh_normal = 0.01
    w_laplacian = 0  # 0.1
    rgb_losses = []
    depth_losses = []
    normals_losses = []
    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    mesh_normal_losses = []
    loop = tqdm(range(n_iter))

    for init_name, dfm_verts in init_verts.items():

        optimizer = torch.optim.Adam([dfm_verts], lr=0.001)

        for i in loop:

            dfm_mesh = tplt_mesh.offset_verts(dfm_verts)

            dfm_render = camera.render(dfm_mesh.extend(len(elevation)))
            diff_rgb = torch.abs(
                tgt_render["rgb"].float() - dfm_render["rgb"].float()
            )
            diff_depth = torch.abs(
                tgt_render["depth"].float() - dfm_render["depth"].float()
            )
            diff_normals = torch.abs(
                tgt_render["normals"].float() - dfm_render["normals"].float()
            )
            if mask_tgt:
                mask = tgt_render["depth"] > 0
                diff_rgb *= mask
                diff_depth *= mask
                diff_normals *= mask
            loss_rgb = diff_rgb.mean()
            loss_depth = diff_depth.mean()
            loss_normals = diff_normals.mean()

            tgt_sample = sample_points_from_meshes(tgt_mesh, 5000)
            dfm_sample = sample_points_from_meshes(dfm_mesh, 5000)

            loss_chamfer, _ = chamfer_distance(tgt_sample, dfm_sample)
            loss_edge = mesh_edge_loss(dfm_mesh)
            loss_mesh_normal = mesh_normal_consistency(dfm_mesh)
            loss_laplacian = mesh_laplacian_smoothing(
                dfm_mesh, method="uniform"
            )

            offset_loss = torch.mean(
                ((tgt_mesh.verts_packed() - dfm_mesh.verts_packed()) ** 2)
                .sum(-1)
                .sqrt()
            )

            rgb_losses.append(float(loss_rgb.detach().cpu()))
            depth_losses.append(float(loss_depth.detach().cpu()))
            normals_losses.append(float(loss_normals.detach().cpu()))
            chamfer_losses.append(float(loss_chamfer.detach().cpu()))
            edge_losses.append(float(loss_edge.detach().cpu()))
            mesh_normal_losses.append(float(loss_mesh_normal.detach().cpu()))
            laplacian_losses.append(float(loss_laplacian.detach().cpu()))
            loss = (
                loss_rgb * w_rgb
                + loss_depth * w_depth
                + loss_normals * w_normals
                + loss_chamfer * w_chamfer
                + loss_edge * w_edge
                + loss_mesh_normal * w_mesh_normal
                + loss_laplacian * w_laplacian
            )
            loop.set_description(
                f"Optimization loss = {loss:.6f}, Offset loss = {offset_loss:.6f}"
            )
            if i % plot_period == 0:
                if (i == 0) and (init_name == "no_init"):
                    continue
                # plot_pointcloud(dfm_mesh, title=f"iter: {i}")
                save_prefix = f"vis/obj{obj_name}/{init_name}_{i}"
                with torch.no_grad():
                    vis_render = camera.render(
                        dfm_mesh.extend(len(elevation)),
                        vis=False,
                        save_prefix=save_prefix,
                    )
                print(f"Rendered image saved to {save_prefix}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
