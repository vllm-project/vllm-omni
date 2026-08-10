# ruff: noqa
import torch


def create_grid_mask(seq_length, trunck_length, fill_triangle):
    assert seq_length > 0

    # 先不考虑seen_length创建一个grid mask：
    if fill_triangle:
        mask = 1 - torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        # 下三角与主对角线都为1
    else:
        mask = torch.zeros(seq_length, seq_length)

    for i in range(seq_length):
        trunck_idx = i // trunck_length
        trunck_start = trunck_idx * trunck_length
        trunck_end = trunck_length + trunck_start
        mask[i][trunck_start:trunck_end] = 1

    return mask


if __name__ == "__main__":
    mask = create_grid_mask(seq_length=8, trunck_length=3, fill_triangle=True).int()
    print(mask)
# tensor([[1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1]]
