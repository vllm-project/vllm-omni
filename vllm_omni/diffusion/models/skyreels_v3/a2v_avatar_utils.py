import torch
from einops import rearrange

try:
    import kornia
except ImportError:
    kornia = None

try:
    from xfuser.core.distributed import get_sp_group
except ImportError:

    def get_sp_group():
        raise RuntimeError(
            "SkyReels V3 A2V sequence parallelism requires the optional xfuser package."
        )

# fmt: off
ASPECT_RATIO_627 = {
     '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.577': ([480, 832], 1), '0.67': ([512, 768], 1), 
     '0.82': ([576, 704], 1),  '1.00': ([640, 640], 1),  '1.22': ([704, 576], 1), '1.50': ([768, 512], 1), 
     '1.86': ([832, 448], 1),  '1.73': ([832, 480], 1),  '2.00': ([896, 448], 1),  '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1), 
     '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)}


ASPECT_RATIO_960 = {
     '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1), 
     '0.55': ([704, 1280], 1), '0.56': ([720, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1), 
     '1.00': ([960, 960], 1), '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1), 
     '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.78': ([1280, 720], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1), 
     '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1), 
     '3.75': ([1920, 512], 1)}
# fmt: on


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):

    source_min, source_max = source_range
    new_min, new_max = target_range

    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


# @torch.compile
def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, mode="mean", attn_bias=None):

    ref_k = ref_k.to(visual_q.dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1)  # B, H, x_seqlens, ref_seqlens

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(visual_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(visual_q.dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        # torch_gc()
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = (
            x_ref_attnmap.sum(-1) / ref_target_mask.sum()
        )  # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1)  # B, x_seqlens, H

        if mode == "mean":
            x_ref_attnmap = x_ref_attnmap.mean(-1)  # B, x_seqlens
        elif mode == "max":
            x_ref_attnmap = x_ref_attnmap.max(-1)  # B, x_seqlens

        x_ref_attn_maps.append(x_ref_attnmap)

    del attn
    del x_ref_attn_map_source
    # torch_gc()

    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=2, enable_sp=False):
    """Args:
    query (torch.tensor): B M H K
    key (torch.tensor): B M H K
    shape (tuple): (N_t, N_h, N_w)
    ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape
    if enable_sp:
        ref_k = get_sp_group().all_gather(ref_k, dim=1)

    x_seqlens = N_h * N_w
    ref_k = ref_k[:, :x_seqlens]
    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)

    split_chunk = heads // split_num

    for i in range(split_num):
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(
            visual_q[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_k[:, :, i * split_chunk : (i + 1) * split_chunk, :],
            ref_target_masks,
        )
        x_ref_attn_maps += x_ref_attn_maps_perhead

    return x_ref_attn_maps / split_num


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def match_and_blend_colors(
    source_chunk: torch.Tensor,  # (1, C, T, H, W), range [-1, 1]
    reference_image: torch.Tensor,  # (1, C, 1, H, W), range [-1, 1]
    strength: float,
) -> torch.Tensor:
    if strength == 0.0 or kornia is None:
        return source_chunk

    # shapes
    B, C, T, H, W = source_chunk.shape
    device = source_chunk.device
    input_dtype = source_chunk.dtype  # e.g. bf16/fp16; we会在颜色空间转换时临时转fp32

    with torch.no_grad():
        # [-1,1] -> [0,1]
        src_01 = (source_chunk + 1.0) * 0.5
        ref_01 = (reference_image + 1.0) * 0.5

        # Kornia在fp32上最快最稳；避免fp64
        src32 = src_01.to(torch.float32)
        ref32 = ref_01.to(torch.float32)

        # 将时间维合并进batch，一次性做Lab转换
        # (B, C, T, H, W) -> (B*T, C, H, W)
        src_bt = src32.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        # 参考图只有1帧：(B, C, 1, H, W) -> (B, C, H, W)
        ref_bchw = ref32[:, :, 0, :, :].contiguous()

        # RGB->Lab（批量）
        src_lab = kornia.color.rgb_to_lab(src_bt)  # (B*T, C, H, W)
        ref_lab = kornia.color.rgb_to_lab(ref_bchw)  # (B,   C, H, W)

        # 展平到像素维
        src_lab_flat = src_lab.view(B * T, C, -1)  # (B*T, C, HW)
        ref_lab_flat = ref_lab.view(B, C, -1)  # (B,   C, HW)

        # 单次遍历拿到std和mean，unbiased=False更快也更符合图像统计直觉
        src_std, src_mean = torch.std_mean(src_lab_flat, dim=-1, keepdim=True, unbiased=False)
        ref_std, ref_mean = torch.std_mean(ref_lab_flat, dim=-1, keepdim=True, unbiased=False)

        # 防除零
        src_std = src_std.clamp_min_(1e-6)

        # 将参考统计广播到每一帧
        ref_mean_bt = ref_mean.repeat_interleave(T, dim=0)  # (B*T, C, 1)
        ref_std_bt = ref_std.repeat_interleave(T, dim=0)  # (B*T, C, 1)

        # Lab空间的线性颜色迁移
        corrected_lab_flat = (src_lab_flat - src_mean) * (ref_std_bt / src_std) + ref_mean_bt
        corrected_lab = corrected_lab_flat.view(B * T, C, H, W)

        # Lab->RGB
        corrected_rgb_01 = kornia.color.lab_to_rgb(corrected_lab)  # (B*T, C, H, W)

        # 与原图在[0,1]空间混合
        blended_rgb_01 = (1.0 - strength) * src_bt + strength * corrected_rgb_01

        # 还原回 (B, C, T, H, W)
        blended_rgb_01 = blended_rgb_01.view(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # [0,1] -> [-1,1] 并恢复输入dtype
        out = (blended_rgb_01 * 2.0 - 1.0).to(dtype=input_dtype)

    return out


def process_video_samples(gen_video_samples: torch.Tensor) -> torch.Tensor:
    """
    将模型生成的视频张量从 [-1, 1] 映射到 [0, 255] 像素值，并调整维度顺序以便写出为视频帧。

    Args:
        gen_video_samples: (1, C, T, H, W), range [-1, 1]

    Returns:
        (1, C, T, H, W), dtype torch.uint8, range [0, 255], device: cpu
    """

    processed_video_samples = (gen_video_samples + 1) * 127.5  # [-1,1] -> [0,255]
    processed_video_samples = processed_video_samples.clamp(0, 255).to(torch.uint8).cpu()
    return processed_video_samples
