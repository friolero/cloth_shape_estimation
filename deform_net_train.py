import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils import (
    compute_vertex_normals,
    depth2disparity,
    get_adjacency_matrix,
    get_riemannian_metric,
    load_wavefront_file,
    rayleigh_quotient_curvature,
)
from deform_dataset import DeformDataset
from deform_net import DeformNet
from differentiable_rendering import CameraInterface, init_lighting

# from graph_conv_deform_net import GraphConvDeformNet

if __name__ == "__main__":

    exp_name = sys.argv[1]
    vis_dir = f"vis/{exp_name}"
    output_dir = f"output/{exp_name}"
    log_dir = f"log/{exp_name}"
    if not os.path.isdir(vis_dir):
        os.system(f"mkdir -p {vis_dir}")
    if not os.path.isdir(output_dir):
        os.system(f"mkdir -p {output_dir}")
    if os.path.isdir(log_dir):
        os.system(f"rm -rf {log_dir}")
    writer = SummaryWriter(log_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    n_epoch = 100
    batch_size = 9

    # create data loader
    data_dir = "/home/zyuwei/Projects/cloth_shape_estimation/data"
    cano_obj_fn = f"{data_dir}/textured_flat_cloth.obj"
    cano_verts, _, _, cano_mesh = load_wavefront_file(cano_obj_fn, device)
    adjacency_mtx = get_adjacency_matrix(cano_mesh)
    verts_uv = cano_mesh.textures.verts_uvs_list()[0]
    train_ds = DeformDataset(data_dir=data_dir, mode="train")
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    eval_ds = DeformDataset(data_dir=data_dir, mode="eval")
    eval_dl = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # create differentiable rendering system
    image_size = 256
    cam_dist = 8.0
    elevation = [30.0 for _ in range(batch_size)]
    azimuth = [45.0 for _ in range(batch_size)]
    lights = init_lighting(mode="point", device=device)
    cano_mesh = cano_mesh.extend(batch_size)
    camera = CameraInterface(
        device,
        image_size,
        cam_dist,
        elevation,
        azimuth,
        lights,
        mode=["rgb", "depth", "normals"],
    )

    # initialize model, criterion, lr scheduler and optimizer
    lr = 1e-4
    betas = (0.9, 0.999)
    lr_scheduler_patience = 3
    model = DeformNet(
        cano_verts,
        use_depth=True,
        embed_depth=True,
        use_normals=True,
        embed_normals=False,  # True,
        embed_uv=True,
        c_dim=256,
    )
    # model = GraphConvDeformNet(
    #    cano_verts, adjacency_mtx, use_depth=True, use_normals=True
    # )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=lr_scheduler_patience,
    )
    w_rgb = 0.0  # degenerate -> not used
    w_depth = 0.0  # degenerate -> not used
    w_normals = 0.0  # degenerate -> not used
    w_laplacian = 0.0  # opposite optimization goal -> not used
    w_chamfer = 0.0  # 1.0 not helpful -> not used
    w_offset = 1
    w_mesh_normal = 0.3
    w_mesh_curv = 1000
    w_riemannian_reg = 0  # 100
    writer.add_hparams(
        {
            "lr": lr,
            "betas": torch.Tensor(betas),
            "lr_scheduler_patience": lr_scheduler_patience,
            "w_rgb": w_rgb,
            "w_depth": w_depth,
            "w_normals": w_normals,
            "w_chamfer": w_chamfer,
            "w_offset": w_offset,
            "w_laplacian": w_laplacian,
            "w_meshnormal": w_mesh_normal,
            "w_meshcurvature": w_mesh_curv,
            "w_riemannian_reg": w_riemannian_reg,
        },
        dict(),
    )

    n_log = 1
    vis = False
    best_loss = 1e9
    for epoch_idx in range(n_epoch):
        print(f"====================Epoch {epoch_idx}====================")
        # training
        n_data = 0
        total_loss = {
            "weighted": 0,
            "rgb": 0,
            "depth": 0,
            "normals": 0,
            "chamfer": 0,
            "offset": 0,
            "laplacian": 0,
            "mesh_normal": 0,
            "mesh_curvature": 0,
            "riemannian_reg": 0,
        }
        for idx, batch in enumerate(tqdm(train_dl)):

            # if idx >= 10:
            #    break

            model.train()
            optimizer.zero_grad()

            # model forwarding
            batch = [data.to(device) for data in batch]
            rgb, depth, normals, offsets = batch
            disparity = depth2disparity(depth)
            mask = depth > 0
            gt_mesh = cano_mesh.offset_verts(offsets.reshape(-1, 3))

            pred_dfm_vtx, pred_offsets = model(
                rgb,
                disparity,
                normals,
                verts_uv,
            )

            # differentiable rendering the predicted mesh
            dfm_mesh = cano_mesh.offset_verts(pred_offsets.reshape(-1, 3))
            tgt_render = camera.render(dfm_mesh)
            if idx == 0:
                with torch.no_grad():
                    gt_render = camera.render(gt_mesh)
                vis_images = []
                for mode in camera.mode:
                    pred = torch.cat(
                        [
                            img.permute([2, 0, 1])
                            for img in tgt_render[mode][..., :3]
                        ],
                        2,
                    )
                    gt = torch.cat(
                        [
                            img.permute([2, 0, 1])
                            for img in gt_render[mode][..., :3]
                        ],
                        2,
                    )
                    vis_img = torch.cat([gt, pred], 1)
                    if mode == "depth":
                        vis_img = (vis_img - vis_img.min()) / (
                            vis_img.max() - vis_img.min()
                        )
                        vis_img = vis_img.repeat([3, 1, 1])
                    if mode == "normals":
                        vis_img = ((vis_img + 1.0) / 2)[[0, 2, 1]]
                        vis_mask = ~vis_img.eq(
                            torch.Tensor([0.5, 0.5, 0.5])
                            .reshape(-1, 1, 1)
                            .to(vis_img.device)
                        )
                        vis_img *= vis_mask
                    vis_images.append(vis_img)
                writer.add_image(
                    f"train/images", torch.cat(vis_images, 1), epoch_idx
                )
                del gt_render, vis_images, vis_img
            pred_rgb = tgt_render["rgb"][..., :3].permute([0, 3, 1, 2])
            pred_depth = (
                tgt_render["depth"].permute([0, 3, 1, 2]).repeat([1, 3, 1, 1])
            )
            pred_normals = tgt_render["normals"].permute([0, 3, 1, 2])

            # loss calculation
            rgb_loss = torch.mean(torch.abs(rgb - pred_rgb) * mask)
            depth_loss = torch.mean(torch.abs(depth - pred_depth) * mask)
            # normals_loss = torch.mean(torch.abs(normals - pred_normals) * mask)
            normals_loss = torch.mean(
                (1 - F.cosine_similarity(normals, pred_normals, dim=1)).pow(2)
                * mask[:, 0, :, :]
            )

            gt_sample = sample_points_from_meshes(gt_mesh, 5000)
            dfm_sample = sample_points_from_meshes(dfm_mesh, 5000)
            chamfer_loss, _ = chamfer_distance(gt_sample, dfm_sample)

            offset_loss = torch.mean(
                ((offsets - pred_offsets) ** 2).sum(-1).sqrt()
            )
            neighbour_offsets = torch.matmul(
                adjacency_mtx.float(), pred_offsets
            )
            laplacian_loss = torch.mean(
                (neighbour_offsets - pred_offsets) ** 2
                * pred_offsets.shape[1]
                * 3
            )
            pred_verts_normal = dfm_mesh.verts_normals_packed().reshape(
                offsets.shape[0], -1, 3
            )
            verts_normal = gt_mesh.verts_normals_packed().reshape(
                offsets.shape[0], -1, 3
            )
            mesh_normal_loss = torch.mean(
                (
                    1
                    - F.cosine_similarity(
                        pred_verts_normal, verts_normal, dim=-1
                    )
                ).pow(2)
            )
            pred_min_rq_curv, pred_max_rq_curv = rayleigh_quotient_curvature(
                dfm_mesh, adjacency_mtx
            )
            gt_min_rq_curv, gt_max_rq_curv = rayleigh_quotient_curvature(
                gt_mesh, adjacency_mtx
            )
            mesh_curv_loss = torch.mean(
                (gt_min_rq_curv - pred_min_rq_curv).pow(2)
                + (gt_max_rq_curv - pred_max_rq_curv).pow(2)
            )

            dfm_g = get_riemannian_metric(dfm_mesh)
            gt_g = get_riemannian_metric(gt_mesh)
            riemannian_reg_loss = torch.mean((dfm_g - gt_g).pow(2))

            loss = (
                w_rgb * rgb_loss
                + w_depth * depth_loss
                + w_normals * normals_loss
                + w_chamfer * chamfer_loss
                + w_offset * offset_loss
                + w_laplacian * laplacian_loss
                + w_mesh_normal * mesh_normal_loss
                + w_mesh_curv * mesh_curv_loss
                + w_riemannian_reg * riemannian_reg_loss
            )
            # backwarding loss
            loss.backward()
            optimizer.step()

            n_batch_data = offsets.shape[0]
            total_loss["weighted"] += loss.item() * n_batch_data
            total_loss["rgb"] += rgb_loss.item() * n_batch_data
            total_loss["depth"] += depth_loss.item() * n_batch_data
            total_loss["normals"] += normals_loss.item() * n_batch_data
            total_loss["chamfer"] += chamfer_loss.item() * n_batch_data
            total_loss["offset"] += offset_loss.item() * n_batch_data
            total_loss["laplacian"] += laplacian_loss.item() * n_batch_data
            total_loss["mesh_normal"] += mesh_normal_loss.item() * n_batch_data
            total_loss["mesh_curvature"] += mesh_curv_loss.item() * n_batch_data
            total_loss["riemannian_reg"] += (
                riemannian_reg_loss.item() * n_batch_data
            )
            n_data += n_batch_data

            if idx % n_log == 0:
                step = len(train_dl) * epoch_idx + idx
                writer.add_scalar("train/weighted_loss", loss, step)
                writer.add_scalar("train/rgb_loss", rgb_loss.item(), step)
                writer.add_scalar("train/depth_loss", depth_loss.item(), step)
                writer.add_scalar(
                    "train/normals_loss", normals_loss.item(), step
                )
                writer.add_scalar(
                    "train/chamfer_loss", chamfer_loss.item(), step
                )
                writer.add_scalar("train/offset_loss", offset_loss.item(), step)
                writer.add_scalar(
                    "train/laplacian_loss", laplacian_loss.item(), step
                )
                writer.add_scalar(
                    "train/mesh_normal_loss", mesh_normal_loss.item(), step
                )
                writer.add_scalar(
                    "train/mesh_curvature_loss", mesh_curv_loss.item(), step
                )
                writer.add_scalar(
                    "train/riemannian_reg", riemannian_reg_loss.item(), step
                )
        for k, v in total_loss.items():
            writer.add_scalar(f"loss/{k}/train", v / n_data, epoch_idx)

        n_data = 0
        total_loss = {
            "weighted": 0,
            "rgb": 0,
            "depth": 0,
            "normals": 0,
            "chamfer": 0,
            "offset": 0,
            "laplacian": 0,
            "mesh_normal": 0,
            "mesh_curvature": 0,
            "riemannian_reg": 0,
        }
        for idx, batch in enumerate(tqdm(eval_dl)):
            # if idx >= 10:
            #    break

            model.eval()
            with torch.no_grad():
                # model forwarding
                batch = [data.to(device) for data in batch]
                rgb, depth, normals, offsets = batch
                disparity = depth2disparity(depth)
                mask = depth > 0
                gt_mesh = cano_mesh.offset_verts(offsets.reshape(-1, 3))
                pred_dfm_vtx, pred_offsets = model(
                    rgb,
                    disparity,
                    normals,
                    verts_uv,
                )

                # differentiable rendering the predicted mesh
                dfm_mesh = cano_mesh.offset_verts(pred_offsets.reshape(-1, 3))
                if idx == 0:
                    tgt_render = camera.render(
                        dfm_mesh,
                        vis=vis,
                        save_prefix=f"vis/eval_pred_e{epoch_idx}",
                    )
                    gt_mesh = cano_mesh.offset_verts(offsets.reshape(-1, 3))
                    gt_render = camera.render(
                        gt_mesh,
                        vis=vis,
                        save_prefix=f"vis/eval_gt_e{epoch_idx}",
                    )
                    vis_images = []
                    for mode in camera.mode:
                        pred = torch.cat(
                            [
                                img.permute([2, 0, 1])
                                for img in tgt_render[mode][..., :3]
                            ],
                            2,
                        )
                        gt = torch.cat(
                            [
                                img.permute([2, 0, 1])
                                for img in gt_render[mode][..., :3]
                            ],
                            2,
                        )
                        vis_img = torch.cat([gt, pred], 1)
                        if mode == "depth":
                            vis_img = (vis_img - vis_img.min()) / (
                                vis_img.max() - vis_img.min()
                            )
                            vis_img = vis_img.repeat([3, 1, 1])
                        if mode == "normals":
                            vis_img = ((vis_img + 1.0) / 2)[[0, 2, 1]]
                            vis_mask = ~vis_img.eq(
                                torch.Tensor([0.5, 0.5, 0.5])
                                .reshape(-1, 1, 1)
                                .to(vis_img.device)
                            )
                            vis_img *= vis_mask
                        vis_images.append(vis_img)
                    writer.add_image(
                        f"eval/images", torch.cat(vis_images, 1), epoch_idx
                    )
                    del gt_render, vis_images, vis_img
                else:
                    tgt_render = camera.render(dfm_mesh)
                pred_rgb = tgt_render["rgb"][..., :3].permute([0, 3, 1, 2])
                pred_depth = (
                    tgt_render["depth"]
                    .permute([0, 3, 1, 2])
                    .repeat([1, 3, 1, 1])
                )
                pred_normals = tgt_render["normals"].permute([0, 3, 1, 2])

                # loss calculation
                rgb_loss = torch.mean(torch.abs(rgb - pred_rgb) * mask)
                depth_loss = torch.mean(torch.abs(depth - pred_depth) * mask)
                # normals_loss = torch.mean(
                #    torch.abs(normals - pred_normals) * mask
                # )
                normals_loss = torch.mean(
                    (1 - F.cosine_similarity(normals, pred_normals, dim=1)).pow(
                        2
                    )
                    * mask[:, 0, :, :]
                )
                gt_sample = sample_points_from_meshes(gt_mesh, 5000)
                dfm_sample = sample_points_from_meshes(dfm_mesh, 5000)
                chamfer_loss, _ = chamfer_distance(gt_sample, dfm_sample)

                offset_loss = torch.mean(
                    ((offsets - pred_offsets) ** 2).sum(-1).sqrt()
                )
                neighbour_offsets = torch.matmul(
                    adjacency_mtx.float(), pred_offsets
                )
                laplacian_loss = torch.mean(
                    (neighbour_offsets - pred_offsets) ** 2
                    * pred_offsets.shape[1]
                    * 3
                )

                pred_verts_normal = dfm_mesh.verts_normals_packed().reshape(
                    offsets.shape[0], -1, 3
                )
                verts_normal = gt_mesh.verts_normals_packed().reshape(
                    offsets.shape[0], -1, 3
                )
                mesh_normal_loss = torch.mean(
                    (
                        1
                        - F.cosine_similarity(
                            pred_verts_normal, verts_normal, dim=-1
                        )
                    ).pow(2)
                )

                (
                    pred_min_rq_curv,
                    pred_max_rq_curv,
                ) = rayleigh_quotient_curvature(dfm_mesh, adjacency_mtx)
                gt_min_rq_curv, gt_max_rq_curv = rayleigh_quotient_curvature(
                    gt_mesh, adjacency_mtx
                )
                mesh_curv_loss = torch.mean(
                    (gt_min_rq_curv - pred_min_rq_curv).pow(2)
                    + (gt_max_rq_curv - pred_max_rq_curv).pow(2)
                )

                dfm_g = get_riemannian_metric(dfm_mesh)
                gt_g = get_riemannian_metric(gt_mesh)
                riemannian_reg_loss = torch.mean((dfm_g - gt_g).pow(2))

                loss = (
                    w_rgb * rgb_loss
                    + w_depth * depth_loss
                    + w_normals * normals_loss
                    + w_chamfer * chamfer_loss
                    + w_offset * offset_loss
                    + w_laplacian * laplacian_loss
                    + w_mesh_normal * mesh_normal_loss
                    + w_mesh_curv * mesh_curv_loss
                    + w_riemannian_reg * riemannian_reg_loss
                )
                scheduler.step(loss)
                n_batch_data = offsets.shape[0]
                total_loss["weighted"] += loss.item() * n_batch_data
                total_loss["rgb"] += rgb_loss.item() * n_batch_data
                total_loss["depth"] += depth_loss.item() * n_batch_data
                total_loss["normals"] += normals_loss.item() * n_batch_data
                total_loss["chamfer"] += chamfer_loss.item() * n_batch_data
                total_loss["offset"] += offset_loss.item() * n_batch_data
                total_loss["laplacian"] += laplacian_loss.item() * n_batch_data
                total_loss["mesh_normal"] += (
                    mesh_normal_loss.item() * n_batch_data
                )
                total_loss["mesh_curvature"] += (
                    mesh_curv_loss.item() * n_batch_data
                )
                total_loss["riemannian_reg"] += (
                    riemannian_reg_loss.item() * n_batch_data
                )

                n_data += n_batch_data

        eval_loss = total_loss["weighted"] / n_data
        for k, v in total_loss.items():
            writer.add_scalar(f"loss/{k}/eval", v / n_data, epoch_idx)
        if best_loss > eval_loss:
            ckpt_fn = f"{output_dir}/model_e{epoch_idx}_{eval_loss:.4f}.pt"
            os.system(f"rm {output_dir}/*.pt")
            torch.save(
                {
                    "epoch": epoch_idx,
                    "loss": eval_loss,
                    "optimizer": optimizer.state_dict(),
                    "model": model.state_dict(),
                },
                ckpt_fn,
            )
            best_loss = eval_loss
            print(f"Saving the model to {ckpt_fn}")
