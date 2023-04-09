import glob
import os
import pickle as pkl
import random
import sys

import numpy as np
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from torchvision.utils import save_image
from tqdm import tqdm


def load_wavefront_file(obj_fn, device, offset=[0.0, 0.0, 0.0], scale=1):
    verts, faces, aux = load_obj(obj_fn, device=device)
    verts = verts.to(device)
    verts = (verts * scale) + torch.Tensor(offset).to(device)
    face_idxs = faces.verts_idx.to(device)
    tex_map = aux.texture_images
    if tex_map is not None and len(tex_map) > 0:
        verts_uvs = aux.verts_uvs.to(device)  # V, 2
        faces_uvs = faces.textures_idx.to(device)  # V, 2
        image = list(tex_map.values())[0].to(device)[None]
        tex = TexturesUV(
            verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
        )
    else:
        tex = None
    mesh = Meshes(verts=[verts], faces=[face_idxs], textures=tex)
    return verts, face_idxs, tex, mesh


def get_angle_diff(normals, pred_normals):
    inner_prod = (normals * pred_normals).sum(dim=1, keepdim=True)
    gt_norm = normals.pow(2).sum(dim=1, keepdim=True).pow(0.5)
    pred_norm = pred_normals.pow(2).sum(dim=1, keepdim=True).pow(0.5)
    cos_angles = inner_prod / (gt_norm * pred_norm)
    angle_diff = torch.acos(cos_angles)
    return angle_diff


def get_adjacency_matrix(mesh):
    edges = mesh.detach().edges_packed().cpu()
    adjacency_matrix = torch.zeros((edges.max() + 1, edges.max() + 1)).bool()
    adjacency_matrix[edges[:, 0], edges[:, 1]] = 1
    adjacency_matrix[edges[:, 1], edges[:, 0]] = 1
    adjacency_matrix = adjacency_matrix.to(mesh.device)
    return adjacency_matrix


def compute_vertex_normals(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, faces_packed[:, 0], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 1], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 2], faces_normals)

    return torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)


def batch_scale_mesh(cano_verts, cano_faces, tex, batch_scale):
    batch_size = batch_scale.shape[0]
    batch_verts = [cano_verts * scale[0] for scale in batch_scale]
    batch_faces = [cano_faces] * batch_size
    batch_tex = tex.extend(batch_size)
    return Meshes(verts=batch_verts, faces=batch_faces, textures=batch_tex)


def get_vertex_covariance(meshes, adjacency_matrix, packed=True):
    n_verts = adjacency_matrix.shape[0]
    verts = meshes.verts_packed().reshape(-1, n_verts, 3)  # B, N, 1
    mask = torch.ones_like(verts[..., :1])  # B, N, 1
    batch_adj_matrix = adjacency_matrix.unsqueeze(0).float()  # 1, N, N
    nb_sum = torch.matmul(batch_adj_matrix, verts)  # B, N, 3
    count = torch.matmul(batch_adj_matrix, mask)  # B, N, 1
    nb_mean = nb_sum / count  # B, N, 3
    edges = meshes.edges_packed()  # BxE, 3
    e0 = torch.cat([edges[:, 0], edges[:, 1]], dim=0)
    e1 = torch.cat([edges[:, 1], edges[:, 0]], dim=0)
    nb_verts = verts.reshape(-1, 3)[e1]
    diff = nb_verts - nb_mean.reshape(-1, 3)[e0]
    diff = diff.unsqueeze(-1)  # B*E, 3
    cov = torch.matmul(diff, diff.transpose(1, 2))  # B*E, 3, 3
    verts_cov = torch.sparse_coo_tensor(
        e0.unsqueeze(0),
        cov,
        size=(verts.shape[0] * verts.shape[1], 3, 3),
    ).coalesce()
    verts_cov = verts_cov.to_dense()  # B*N, 3, 3
    verts_cov /= count.reshape(-1, 1, 1)  # taking average over neighbours
    if packed:
        return verts_cov  # B*N, 3, 3
    else:
        return verts_cov.reshape(verts.shape[0], verts.shape[1], 3, 3)


def rayleigh_quotient_curvature(meshes, adjacency_matrix):
    """RQ Curvature according to Garnet++
        https://arxiv.org/pdf/2007.10867v1.pdf.

    rq_curvature = g_T * cov * g / g_T * g

    Arguments:
        verts: vertex input (batch * N, 3)
        verts_cov: vertex covariance(batch * N, 3, 3)
    """
    verts = meshes.verts_packed()
    verts_cov = get_vertex_covariance(meshes, adjacency_matrix, packed=True)

    rq_curv = torch.matmul(
        torch.matmul(verts.unsqueeze(1), verts_cov), verts.unsqueeze(-1)
    ) / torch.matmul(verts.unsqueeze(1), verts.unsqueeze(-1))
    rq_curv = rq_curv.reshape(-1, adjacency_matrix.shape[0])

    nb_rq_curv = (
        rq_curv.unsqueeze(-1).repeat([1, 1, adjacency_matrix.shape[-1]])
        * adjacency_matrix
    )
    min_rq_curv, _ = nb_rq_curv.min(-1)
    max_rq_curv, _ = nb_rq_curv.max(-1)
    return min_rq_curv, max_rq_curv


def get_riemannian_metric(vertices, faces):
    n_faces = faces.shape[0]
    alpha = torch.zeros((n_faces, 3, 2)).to(
        dtype=faces.dtype, device=faces.device
    )
    V0, V1, V2 = (
        vertices.index_select(0, faces[:, 0]),
        vertices.index_select(0, faces[:, 1]),
        vertices.index_select(0, faces[:, 2]),
    )
    alpha[:, :, 0] = V1 - V0
    alpha[:, :, 1] = V2 - V0
    riemannian_metric = torch.matmul(alpha.transpose(1, 2), alpha)

    return riemannian_metric


def partition_data(in_dir, seed):
    random.seed(seed)

    files = glob.glob(f"{in_dir}/perturbation_mode_0/*.obj")
    files += glob.glob(f"{in_dir}/perturbation_mode_1/*.obj")
    random.shuffle(files)

    for mode in ["train", "eval", "test"]:
        if os.path.isdir(f"data/{mode}"):
            os.system(f"rm -rf data/{mode}")
        os.system(f"mkdir data/{mode}")
        os.system(f"cp {in_dir}/*.jpg data/{mode}")
        os.system(f"cp {in_dir}/*.mtl data/{mode}")

    eval_start_idx = int(0.8 * len(files))
    test_start_idx = int(0.9 * len(files))
    for i, fn in enumerate(files[:eval_start_idx]):
        os.system(f"ln -sf {os.path.abspath(fn)} data/train/{i}.obj")
    for i, fn in enumerate(files[eval_start_idx:test_start_idx]):
        os.system(f"ln -sf {os.path.abspath(fn)} data/eval/{i}.obj")
    for i, fn in enumerate(files[test_start_idx:]):
        os.system(f"ln -sf {os.path.abspath(fn)} data/test/{i}.obj")


def mesh2image(
    device,
    camera,
    obj_files,
    cano_obj_fn,
    max_scale=1,
    min_scale=1,
    max_offset=0,
    min_offset=0,
    n_image_per_obj=1,
):

    cano_verts, _, _, _ = load_wavefront_file(cano_obj_fn, device)
    for obj_fn in tqdm(obj_files):
        for idx in range(n_image_per_obj):
            scale = torch.rand(1) * (max_scale - min_scale) + min_scale
            scale = scale.to(device)
            offset = torch.rand(3) * (max_offset - min_offset) + min_offset
            offset = offset.to(device)
            verts, face_idxs, tex, mesh = load_wavefront_file(
                obj_fn, device, offset=offset, scale=scale
            )

            offsets = verts - cano_verts * scale
            images = camera.render(mesh.extend(1))
            prefix = obj_fn.replace(".obj", f"_{idx}")
            with open(f"{prefix}_offset.pkl", "wb") as fp:
                pkl.dump(
                    {
                        # "rgb": images["rgb"].detach().cpu().numpy(),
                        "scale": scale,
                        "depth": images["depth"].detach().cpu()[0],
                        # "normals": images["normals"].detach().cpu().numpy(),
                        "offsets": offsets.cpu(),
                    },
                    fp,
                    protocol=2,
                )
            save_image(
                images["rgb"].detach().cpu().permute([0, 3, 1, 2])[:, :3, :, :],
                f"{prefix}_rgb.png",
            )
            # Note: diffcloth is set in y-up coordinate system
            # normals = (
            #     images["normals"].detach().cpu().permute([0, 3, 1, 2]) + 1.0
            # ) / 2
            # normals = normals[:, [0, 2, 1], :]
            save_image(
                (images["normals"].detach().cpu().permute([0, 3, 1, 2]) + 1.0)
                / 2,
                # normals,
                f"{prefix}_normals.png",
            )


# generate training data for deform_net
if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    from differentiable_rendering import CameraInterface, init_lighting

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

    # import sys
    # base_dir = sys.argv[1]
    base_dir = "/home/zyuwei/Projects/cloth_shape_estimation/data/"
    cano_obj_fn = f"{base_dir}/textured_flat_cloth.obj"
    for mode in ["eval", "test", "train"]:
        if not os.path.isdir(f"{base_dir}/{mode}"):
            print(f"==> Partition obj data in {base_dir} into train/eval/test")
            partition_data(base_dir, seed=77)
        print(f"==> {mode}")
        obj_files = glob.glob(f"{base_dir}/{mode}/*.obj")
        mesh2image(
            device,
            camera,
            obj_files,
            cano_obj_fn,
        )
