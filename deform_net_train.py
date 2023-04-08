import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils import get_adjacency_matrix, load_wavefront_file
from deform_dataset import DeformDataset
from deform_net import DeformNet
from differentiable_rendering import CameraInterface, init_lighting

if __name__ == "__main__":

    exp_name = sys.argv[1]
    vis_dir = f"vis/{exp_name}"
    output_dir = f"output/{exp_name}"
    log_dir = f"log/{exp_name}"
    if not os.path.isdir(vis_dir):
        os.system(f"mkdir -p {vis_dir}")
    if not os.path.isdir(output_dir):
        os.system(f"mkdir -p {output_dir}")
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
    train_ds = DeformDataset(data_dir=data_dir, mode="test")
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
    cano_mesh = cano_mesh.extend(batch_size)
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

    # initialize model, criterion, lr scheduler and optimizer
    lr = 1e-4
    betas = (0.9, 0.999)
    lr_scheduler_patience = 5
    model = DeformNet(cano_verts).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           betas=betas)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=lr_scheduler_patience,
    )
    w_rgb = 0.0
    w_depth = 0  # 0.1
    w_normals = 0.0
    w_laplacian = 0  # 0
    w_offset = 1  # 0


    vis = False
    best_loss = 1e9
    n_log = 1
    writer.add_hparams({'lr': lr,
                        'betas': torch.Tensor(betas),
                        'lr_scheduler_patience': lr_scheduler_patience,
                        'w_rgb': w_rgb,
                        'w_depth': w_depth,
                        'w_normals': w_normals,
                        'w_offset': w_offset,
                        'w_laplacian': w_laplacian}, dict())
    for epoch_idx in range(n_epoch):
        print(f"====================Epoch {epoch_idx}====================")
        # training
        n_data = 0
        total_loss = {'weighted': 0,
                      'rgb': 0,
                      'depth': 0,
                      'normals': 0,
                      'offset': 0,
                      'laplacian': 0}
        for idx, batch in enumerate(tqdm(train_dl)):
            #if idx >= 10:
            #    break

            model.train()
            optimizer.zero_grad()

            # model forwarding
            batch = [data.to(device) for data in batch]
            rgb, depth, normals, scale, offsets = batch
            mask = depth > 0
            pred_dfm_vtx, pred_offsets, _ = model(scale, rgb, depth, normals)

            # differentiable rendering the predicted mesh
            dfm_mesh = cano_mesh.offset_verts(pred_offsets.reshape(-1, 3))
            tgt_render = camera.render(dfm_mesh)
            if idx == 0:
                with torch.no_grad():
                    gt_mesh = cano_mesh.offset_verts(offsets.reshape(-1, 3))
                    gt_render = camera.render(gt_mesh)
                vis_images = []
                for mode in camera.mode:
                    pred = torch.cat([img.permute([2, 0, 1]) for img in tgt_render[mode][..., :3]], 2)
                    gt = torch.cat([img.permute([2, 0, 1]) for img in gt_render[mode][..., :3]], 2)
                    vis_img = torch.cat([gt, pred], 1)
                    if mode == 'depth':
                        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
                        vis_img = vis_img.repeat([3, 1, 1])
                    if mode == 'normals':
                        vis_img = ((vis_img + 1.0) / 2)[[0, 2, 1]]
                        vis_mask = ~vis_img.eq(torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(vis_img.device))
                        vis_img *= vis_mask
                    vis_images.append(vis_img)
                writer.add_image(f"train/images", torch.cat(vis_images, 1), epoch_idx)
                del gt_mesh, gt_render, vis_images, vis_img
            pred_rgb = tgt_render["rgb"][..., :3].permute([0, 3, 1, 2])
            pred_depth = (
                tgt_render["depth"].permute([0, 3, 1, 2]).repeat([1, 3, 1, 1])
            )
            pred_normals = tgt_render["normals"].permute([0, 3, 1, 2])

            # loss calculation
            rgb_loss = torch.mean(torch.abs(rgb - pred_rgb) * mask)
            depth_loss = torch.mean(torch.abs(depth - pred_depth) * mask)
            normals_loss = torch.mean(torch.abs(normals - pred_normals) * mask)
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
            loss = (
                w_rgb * rgb_loss
                + w_depth * depth_loss
                + w_normals * normals_loss
                + w_offset * offset_loss
                + w_laplacian * laplacian_loss
            )
            # backwarding loss
            loss.backward()
            optimizer.step()

            n_batch_data = offsets.shape[0]
            total_loss['weighted'] += loss.item() * n_batch_data
            total_loss['rgb'] += rgb_loss.item() * n_batch_data
            total_loss['depth'] += depth_loss.item() * n_batch_data
            total_loss['normals'] += normals_loss.item() * n_batch_data
            total_loss['offset'] += offset_loss.item() * n_batch_data
            total_loss['laplacian'] += laplacian_loss.item() * n_batch_data
            n_data += n_batch_data

            if idx % n_log == 0:
                step = len(train_dl) * epoch_idx + idx
                writer.add_scalar('train/weighted_loss', loss, step)
                writer.add_scalar('train/rgb_loss', rgb_loss.item(), step)
                writer.add_scalar('train/depth_loss', depth_loss.item(), step)
                writer.add_scalar('train/normals_loss', normals_loss.item(), step)
                writer.add_scalar('train/offset_loss', offset_loss.item(), step)
                writer.add_scalar('train/laplacian_loss', laplacian_loss.item(), step)
        for k, v in total_loss.items():
            writer.add_scalar(f'loss/{k}/train', v / n_data, epoch_idx)

        n_data = 0
        total_loss = {'weighted': 0,
                      'rgb': 0,
                      'depth': 0,
                      'normals': 0,
                      'offset': 0,
                      'laplacian': 0}
        for idx, batch in enumerate(tqdm(eval_dl)):
            #if idx >= 10:
            #    break

            model.eval()
            with torch.no_grad():
                # model forwarding
                batch = [data.to(device) for data in batch]
                rgb, depth, normals, scale, offsets = batch
                mask = depth > 0
                try:
                    pred_dfm_vtx, pred_offsets, _ = model(
                        scale, rgb, depth, normals
                    )
                except:
                    import ipdb; ipdb.set_trace()

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
                        pred = torch.cat([img.permute([2, 0, 1]) for img in tgt_render[mode][..., :3]], 2)
                        gt = torch.cat([img.permute([2, 0, 1]) for img in gt_render[mode][..., :3]], 2)
                        vis_img = torch.cat([gt, pred], 1)
                        if mode == 'depth':
                            vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min())
                            vis_img = vis_img.repeat([3, 1, 1])
                        if mode == 'normals':
                            vis_img = ((vis_img + 1.0) / 2)[[0, 2, 1]]
                            vis_mask = ~vis_img.eq(torch.Tensor([0.5, 0.5, 0.5]).reshape(-1, 1, 1).to(vis_img.device))
                            vis_img *= vis_mask
                        vis_images.append(vis_img)
                    writer.add_image(f"eval/images", torch.cat(vis_images, 1), epoch_idx)
                    del gt_mesh, gt_render, vis_images, vis_img
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
                normals_loss = torch.mean(
                    torch.abs(normals - pred_normals) * mask
                )
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
                loss = (
                    w_rgb * rgb_loss
                    + w_depth * depth_loss
                    + w_normals * normals_loss
                    + w_offset * offset_loss
                    + w_laplacian * laplacian_loss
                )
                scheduler.step(loss)
                n_batch_data = offsets.shape[0]
                total_loss['weighted'] += loss.item() * n_batch_data
                total_loss['rgb'] += rgb_loss.item() * n_batch_data
                total_loss['depth'] += depth_loss.item() * n_batch_data
                total_loss['normals'] += normals_loss.item() * n_batch_data
                total_loss['offset'] += offset_loss.item() * n_batch_data
                total_loss['laplacian'] += laplacian_loss.item() * n_batch_data
                n_data += n_batch_data

        eval_loss = total_loss['weighted'] / n_data
        for k, v in total_loss.items():
            writer.add_scalar(f'loss/{k}/eval', v / n_data, epoch_idx)
        if best_loss > eval_loss:
            ckpt_fn = f"{output_dir}/model_e{epoch_idx}_{eval_loss:.4f}.pt"
            torch.save(
                {
                    'epoch': epoch_idx,
                    'loss': eval_loss,
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict(),
                }, ckpt_fn)
            best_loss = eval_loss
            print(f"Saving the model to {ckpt_fn}")
