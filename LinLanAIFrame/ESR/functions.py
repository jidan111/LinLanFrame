from .structs import *


def split_image_block(tensor, rows, cols, return_cat=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    channel, high, width = tensor.shape
    h_row, h_div = divmod(high, rows)
    w_col, w_div = divmod(width, cols)
    if return_cat:
        assert h_div == 0 and w_div == 0, f"无法被均分为{rows}行{cols}列"
    out = []
    true_col = 0
    for h in range(0, high, h_row):
        true_col = 0
        for w in range(0, width, w_col):
            out.append(tensor[:, h:h + h_row, w:w + w_col].unsqueeze(0))
            true_col += 1
    if return_cat:
        return torch.cat(out, dim=0).to(device), true_col
    return out, true_col


def combine_image_block(arr, cols):
    row_tensor = []
    for i in range(0, len(arr), cols):
        row_tensor.extend(torch.cat(arr[i:i + cols], dim=3))
    tensor = torch.cat(row_tensor, dim=1)
    return tensor
