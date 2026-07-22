# Adapted from Lingbot-World/wan/utils/cam_utils.py
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    # interpolate translation
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    # interpolate rotation
    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    # ensure there is no sudden change in qw
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(t: torch.Tensor) -> torch.Tensor:
    Rot = t[:, :3, :3]  # [B,3,3]
    trans = t[:, :3, 3:]  # [B,3,1]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=t.device, dtype=t.dtype)[None, :, :].repeat(t.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    # ensure identity matrix for 1st frame
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        # compute pose between i and i+1
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        # scale the coordinate inputs to roughly 1 standard deviation to simplify model learning (camctrl2).
        translations = relative_poses[:, :3, 3]  # [f, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        # only normalize when moving
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(
    n_frames: int, height: int, width: int, bias: float = 0.5, device="cuda", dtype=torch.float32
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias  # [h*w, 2]
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)  # [f, h*w, 2]
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    k: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
):
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)  # [f, h*w, 2]
    fx, fy, cx, cy = k.chunk(4, dim=-1)  # [f, 1]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]  # [f, h*w]
    zs = torch.ones_like(i)  # [f, h*w]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # [f, h*w, 3]
    directions = directions / directions.norm(dim=-1, keepdim=True)  # [f, h*w, 3]

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # [f, h*w, 3]
    if only_rays_d:
        plucker_embeddings = rays_d  # [f, h*w, 3]
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 3])  # [f*h*w, 3]
    else:
        rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
        rays_o = rays_o[:, None, :].expand_as(rays_d)  # [f, h*w, 3]
        plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # [f, h*w, 6]
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6])  # [f*h*w, 6]
    return plucker_embeddings


def get_Ks_transformed(
    k: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    fx, fy, cx, cy = k.chunk(4, dim=-1)  # [f, 1]

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y

    Ks_transformed = torch.zeros_like(k)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed
