import argparse
import contextlib
import os
import sys
from typing import Union

import diffcloth_py as diffcloth
import numpy as np
import torch
from tqdm import tqdm

import common
from pySim.pySim import pySim

parser = argparse.ArgumentParser("Perturb flat cloth")
parser.add_argument(
    "-mode",
    type=int,
    default=1,
    help="-1: no change, 0: random perturb, 1: bezier purturb",
)
parser.add_argument("-r", dest="render", action="store_true", default=False)
parser.add_argument("-s", dest="save", action="store_true", default=True)
parser.add_argument("-task_name", type=str, default="flatten_tshirt")
parser.add_argument("-n_vertices", type=int, default=600)
parser.add_argument("-n_openmp_thread", type=int, default=16)
parser.add_argument("-i", type=int, required=True)
parser.add_argument("-output_dir", type=str, default="cloth_project/")
parser.add_argument("-seed", type=int, default=8824325)
args = parser.parse_args()


class Bezier:
    def parameterized_two_points(
        self, t: Union[int, float], P1: np.ndarray, P2: np.ndarray
    ) -> np.ndarray:
        """Returns a point between P1 and P2, parametised by t."""
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def bezier_points(self, t: Union[int, float], points: np.ndarray) -> list:
        """Returns a list of points interpolated by the Bezier process."""
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [
                self.parameterized_two_points(t, points[i1], points[i1 + 1])
            ]
        return newpoints

    def bezier_point(
        self, t: Union[int, float], points: np.ndarray
    ) -> np.ndarray:
        """Returns a point interpolated by the Bezier process."""
        newpoints = points
        while len(newpoints) > 1:
            newpoints = self.bezier_points(t, newpoints)
        return newpoints[0]

    def bezier_curve(
        self, t_values: Union[tuple, list], points: np.ndarray
    ) -> np.ndarray:
        """Returns a point interpolated by the Bezier process."""
        assert len(t_values) > 0, "t_values must contain at least one value."
        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [self.bezier_point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve


@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


def get_state(sim: diffcloth.Simulation, to_tensor: bool = False) -> tuple:
    state_info_init = sim.getStateInfo()
    x, v = state_info_init.x, state_info_init.v
    clip_pos = np.array(sim.getStateInfo().x_fixedpoints)
    if to_tensor:
        x_t = common.toTorchTensor(x, False, False).clone()
        v_t = common.toTorchTensor(v, False, False).clone()
        a_t = common.toTorchTensor(clip_pos, False, False).clone()
        return x_t, v_t, a_t
    else:
        return x, v, clip_pos


def forward_sim_no_control(
    x_i: torch.Tensor,
    v_i: torch.Tensor,
    a_t: torch.Tensor,
    pysim: pySim,
    steps: int,
) -> list:
    """Pure physics simulation."""
    records = []
    for step in tqdm(range(steps)):
        records.append((x_i, v_i))
        da_t = torch.zeros_like(a_t)
        a_t += da_t
        x_i, v_i = pysim(x_i, v_i, a_t)
    records.append((x_i, v_i))
    return records


def forward_sim_rand_control(
    x_i: torch.Tensor,
    v_i: torch.Tensor,
    a_t: torch.Tensor,
    pysim: pySim,
    steps: int,
    dilution: float = 0.1,
    action_repeat: int = 4,
) -> list:
    records = []
    for step in tqdm(range(steps)):
        records.append((x_i, v_i))
        da_t = (torch.rand(a_t.shape) - 0.5) * 2 * dilution
        if step < (steps / 2):
            da_t[1] = abs(da_t[1]) / 2  # raise up for the first half steps
        else:
            da_t[1] = -abs(da_t[1]) / 2  # lower down for the last half steps
        a_t += da_t
        for _ in range(action_repeat):
            x_i, v_i = pysim(x_i, v_i, a_t)
    records.append((x_i, v_i))
    return records


def get_random_target_within_range(
    x0: torch.Tensor, att_idx: int, min_dist: float, max_dist: float
) -> torch.Tensor:
    reshaped_x0 = x0.reshape(-1, 3)
    fixed_pos = reshaped_x0[att_idx]
    dist = (reshaped_x0 - fixed_pos).norm(dim=-1).numpy()
    mask = np.logical_and(dist > min_dist, dist < max_dist)
    tgt_idx = np.random.choice(np.where(mask)[0])
    tgt_pos = reshaped_x0[tgt_idx]
    tgt_pos[1] += np.random.uniform(0.1, 0.3)
    return tgt_pos


def forward_sim_targeted_control(
    x_i: torch.Tensor,
    v_i: torch.Tensor,
    a_t: torch.Tensor,
    tgt_pos: torch.Tensor,
    pysim: pySim,
    steps: int,
    action_repeat: int = 4,
    min_height: float = 1,
    max_height: float = 3,
) -> list:

    start_pos = a_t.clone().numpy()
    tgt_pos = tgt_pos.numpy()
    mid_high_pos = np.random.uniform(start_pos, tgt_pos)
    mid_high_pos[1] = np.random.uniform(min_height, max_height)
    bezier = Bezier()
    t_points = np.arange(0, 1, 1.0 / steps)
    points = np.array([start_pos, mid_high_pos, tgt_pos])
    curve_points = bezier.bezier_curve(t_points, points)

    records = []
    for points in tqdm(curve_points):
        records.append((x_i, v_i))
        a_t = common.toTorchTensor(points, False, False).clone()
        for _ in range(action_repeat):
            x_i, v_i = pysim(x_i, v_i, a_t)
    records.append((x_i, v_i))
    return records

    return records


def get_center_pos(
    sim: diffcloth.Simulation, corner_idx: list = [315, 314, 284, 285]
) -> torch.Tensor:
    v_pos, _, _ = get_state(sim, to_tensor=True)
    v_pos = v_pos.reshape(-1, 3)
    center_pos = v_pos[torch.LongTensor(corner_idx)].mean(0)
    return center_pos


def export_mesh(
    sim: diffcloth.Simulation,
    out_fn: str,
    tmp_fn: str = "untextured.obj",
    cano_fn: str = "textured_flat_cloth.obj",
    dir_prefix: str = "output/cloth_project",
    export_step: int = None,
    renormalize: bool = False,
) -> None:

    if export_step is None:
        export_step = (sim.sceneConfig.stepNum - 1,)
    # export to untextured object
    sim.exportCurrentMeshPos(
        export_step,
        f"{dir_prefix}/{tmp_fn}".replace("output/", "").replace(".obj", ""),
    )

    if renormalize:
        center_pos = get_center_pos(sim)
        center_pos[2] = 0.0  # in-plane normalization
    if os.path.isfile(f"{dir_prefix}/{tmp_fn}"):
        obj_lines = []
        with open(f"{dir_prefix}/{cano_fn}", "r") as fp:
            found_mtl = False
            cano_vpos_idx = []
            for i, line in enumerate(fp.readlines()):
                obj_lines.append(line)
                if line.startswith("v "):
                    cano_vpos_idx.append(i)
                if ".mtl" in line:
                    found_mtl = True
        if found_mtl:
            with open(f"{dir_prefix}/{tmp_fn}", "r") as fp:
                new_vpos_lines = [
                    line for line in fp.readlines() if line.startswith("v ")
                ]
                assert len(new_vpos_lines) == len(
                    cano_vpos_idx
                ), "the numbers of vertices mismatch"
                for i, line_idx in enumerate(cano_vpos_idx):
                    if renormalize:
                        tmp_pos = new_vpos_lines[i].strip().split()[-3:]
                        pos = [
                            float(n) - center_pos[i]
                            for i, n in enumerate(tmp_pos)
                        ]
                        new_vpos_lines[i] = f"v {pos[0]} {pos[1]} {pos[2]}\n"
                    obj_lines[line_idx] = new_vpos_lines[i]
            with open(f"{dir_prefix}/{out_fn}", "w") as fp:
                fp.write("".join(obj_lines))
            print(f"==> Exported textured obj to {dir_prefix}/{out_fn}")
            os.system(f"rm -f {dir_prefix}/{tmp_fn}")
            os.system(f"rm -f {dir_prefix}/*.txt")
        else:
            print("*********[WARNING]: mtl file not found...*************")
    else:
        print("************[ERROR]: in exporting!!!*************")


def perturb(args, out_fn):
    # Decide fixed point index
    vertex_indices = np.arange(0, args.n_vertices)
    if args.mode >= 0:
        # Corner indices: 0, 29, 570, 599
        att_idx = np.random.choice(vertex_indices)
    else:
        att_idx = -1

    # Initialize the scene with specified fixed point or no fixed point
    helper = diffcloth.makeOptimizeHelper(args.task_name)
    sim = diffcloth.makeSim(
        exampleName=args.task_name,
        runBackward=False,
        customAttachmentVertexIdx=att_idx,
    )
    sim.forwardConvergenceThreshold = 1e-8
    pysim = pySim(sim, helper, True)

    # Reset the system
    sim.resetSystem()
    x0_t, v0_t, a0_t = get_state(sim, to_tensor=True)
    assert (x0_t.shape == v0_t.shape) and (
        x0_t.shape[0] == args.n_vertices * 3
    ), "Wrong number of vertices provided!"

    # Cloth control
    sim.resetSystem()
    if args.mode == -1:
        _ = forward_sim_no_control(x0_t, v0_t, a0_t, pysim, 200)
    else:
        if args.mode == 0:
            _ = forward_sim_rand_control(x0_t, v0_t, a0_t, pysim, 100)
        else:
            tgt_pos = get_random_target_within_range(x0_t, att_idx, 1, 3.5)
            _ = forward_sim_targeted_control(
                x0_t, v0_t, a0_t, tgt_pos, pysim, 100
            )
        # stabilizing the scene
        x_t, v_t, a_t = get_state(sim, to_tensor=True)
        _ = forward_sim_no_control(x_t, v_t, a_t, pysim, 30)

    # Rendering the simulationg
    if args.render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=True)
    # Export final configuration into wavefront file
    if args.save:
        export_mesh(
            sim,
            obj_fn,
            export_step=sim.getStateInfo().stepIdx,
            renormalize=True,
        )
    del sim, pysim


# TODO 1: adding randomize texture ?
# TODO 2: is stabilizing the scene necessary ?
if __name__ == "__main__":

    diffcloth.enableOpenMP(n_threads=args.n_openmp_thread)

    args.seed += args.i
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obj_fn = f"perturbed_cloth_{args.mode}_{args.i}.obj"
    if os.path.isfile(f"output/cloth_project/{obj_fn}"):
        print(f"output/cloth_project/{obj_fn} exists. Skip...")
    else:
        try:
            with stdout_redirected():
                perturb(args, obj_fn)
        except:
            print("Error encountered. Retry....")
