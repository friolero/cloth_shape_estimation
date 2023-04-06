import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import get_adjacency_matrix, load_wavefront_file
from deform_dataset import DeformDataset
from deform_net import DeformNet
from differentiable_rendering import CameraInterface, init_lighting

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    n_epoch = 100
    batch_size = 12

    # create data loader
    data_dir = "/home/zyuwei/Projects/cloth_shape_estimation/data"
    cano_obj_fn = f"{data_dir}/textured_flat_cloth.obj"
    cano_verts, _, _, cano_mesh = load_wavefront_file(cano_obj_fn, device)
    adjacency_mtx = get_adjacency_matrix(cano_mesh)
    train_ds = DeformDataset(data_dir=data_dir, mode="test")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    eval_ds = DeformDataset(data_dir=data_dir, mode="eval")
    eval_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

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
    model = DeformNet(cano_verts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=5,
    )
    w_rgb = 3.0
    w_depth = 0.1
    w_normals = 3.0
    w_laplacian = 0.01

    n_print = 100
    for epoch_idx in range(n_epoch):

        print(f"====================Epoch {epoch_idx}====================")
        # training
        n_data = 0
        total_loss = 0
        for idx, batch in enumerate(tqdm(train_dl)):

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
            pred_rgb = tgt_render["rgb"][..., :3].permute([0, 3, 1, 2])
            pred_depth = (
                tgt_render["depth"].permute([0, 3, 1, 2]).repeat([1, 3, 1, 1])
            )
            pred_normals = tgt_render["normals"].permute([0, 3, 1, 2])

            # loss calculation
            rgb_loss = torch.mean(torch.abs(rgb - pred_rgb) * mask)
            depth_loss = torch.mean(torch.abs(depth - pred_depth) * mask)
            normals_loss = torch.mean(torch.abs(normals - pred_normals) * mask)
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
                + w_laplacian * laplacian_loss
            )
            total_loss += loss.item() * offsets.shape[0]
            n_data += offsets.shape[0]

            # backwarding loss
            loss.backward()
            optimizer.step()

            if idx % n_print == 0:
                print("  mean_train_loss:", total_loss / n_data)
                print("      (rgb_loss):", rgb_loss.item())
                print("      (depth_loss):", depth_loss.item())
                print("      (normals_loss):", normals_loss.item())
                print("      (laplacian_loss):", laplacian_loss.item())

        n_data = 0
        total_loss = 0
        for idx, batch in enumerate(tqdm(eval_dl)):
            model.eval()
            with torch.no_grad():
                # model forwarding
                batch = [data.to(device) for data in batch]
                rgb, depth, normals, scale, offsets = batch
                mask = depth > 0
                pred_dfm_vtx, pred_offsets, _ = model(
                    scale, rgb, depth, normals
                )

                # differentiable rendering the predicted mesh
                dfm_mesh = cano_mesh.offset_verts(pred_offsets.reshape(-1, 3))
                if idx == 1:
                    tgt_render = camera.render(dfm_mesh, vis=True)
                else:
                    tgt_render = camera.render(dfm_mesh, vis=False)
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
                neighbour_offsets = torch.matmul(adjacency_mtx, pred_offsets)
                laplacian_loss = torch.mean(
                    (neighbour_offsets - pred_offsets) ** 2
                    * pred_offsets.shape[1]
                    * 3
                )
                loss = (
                    w_rgb * rgb_loss
                    + w_depth * depth_loss
                    + w_normals * normals_loss
                    + w_laplacian * laplacian_loss
                )
                total_loss += loss.item() * offsets.shape[0]
                n_data += offsets.shape[0]
                scheduler.step(loss)

            if idx % n_print == 0:
                print(f"Step {idx}:")
                print("  mean_eval_loss:", total_loss / n_data)
                print("      (rgb_loss):", rgb_loss.item())
                print("      (depth_loss):", depth_loss.item())
                print("      (normals_loss):", normals_loss.item())
                print("      (laplacian_loss):", laplacian_loss.item())
