import torch

def save_points_obj(pc: torch.Tensor, file_path: str):
    """save point clouds to obj file

    Args:
        pc: [N, 3]
    """
    assert file_path.split(".")[-1] == 'obj'
    pc = pc.cpu()
    with open(file_path, "w") as f:
        for p in pc:
            f.write("v {:f} {:f} {:f}\n".format(p[0], p[1], p[2]))
    