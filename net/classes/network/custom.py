import torch
import torch.autograd as autograd
import torch.nn.functional as F

def grid_sample_2d(input, grid):
    # grid (B, Ho, Wo, 2)
    # input (B, C, H, W)
    # output (B, C, Ho, Wo)
    B, Ho, Wo, _ = grid.shape
    P = Ho * Wo
    _, C, H, W = input.shape

    # grid to index
    reso = torch.tensor([2.0 / (W-1), 2.0 / (H-1)]).view(1, 1, 1, 2).to(device=grid.device)
    # B, Ho, Wo, 2
    grid = (grid+1.0)/reso
    w1h1 = torch.floor(grid).long()
    w2h2 = w1h1 + 1
    assert(w1h1.min() >= 0 and w2h2[..., 0].max() <= W and w2h2[..., 1].max() <= H)

    input = F.pad(input, (0, 1, 0, 1), mode='replicate')

    # B, 4*P: Q11, Q12, Q21, Q22
    index = torch.stack([w1h1[...,1]*(W+1)+w1h1[...,0], (w1h1[...,1]+1)*(W+1)+w1h1[...,0],
                            w1h1[...,1]*(W+1)+w1h1[...,0]+1, (w1h1[...,1]+1)*(W+1)+w1h1[...,0]+1], dim=1).view(B, -1)
    # 4 of (B, C, P) (f11, f12, f21, f22)
    input_4samples = torch.gather(input.view(B, C, -1), -1, index.unsqueeze(1).expand(-1, C, -1)).split(P, dim=-1)

    # B, Ho, Wo, 2 -> B, P
    diff_x2x, diff_y2y = torch.unbind((w2h2 - grid).view(B, -1, 2), dim=-1)
    diff_xx1, diff_yy1 = torch.unbind((grid - w1h1).view(B, -1, 2), dim=-1)

    f11, f12, f21, f22 = input_4samples
    # B,1,P,2 @ B,C,P,2,2
    result = torch.stack([diff_x2x, diff_xx1],dim=-1).reshape(B,1,P,1,2).expand(-1,C,-1,-1,-1).reshape(-1,1,2) @ torch.stack([f11, f12, f21, f22],dim=-1).reshape(B,C,P,2,2).reshape(-1,2,2) @ torch.stack([diff_y2y, diff_yy1],dim=-1).reshape(B,1,P,2,1).expand(-1,C,-1,-1,-1).reshape(-1,2,1)
    result = result.view(B,C,Ho,Wo)

    return result

def grid_sample_3d(input, grid):
    """
    grid (B, Do, Ho, Wo, 3)
    input (B, C, D, H, W)
    output (B, C, Do, Ho, Wo)
    """
    B, Do, Ho, Wo, _ = grid.shape
    P = Ho * Wo * Do
    _, C, D, H, W = input.shape

    # ref = F.grid_sample(input, grid, padding_mode="border", align_corners=True)
    # grid to index B,1,1,1,3
    reso = torch.tensor([2.0/ (D-1), 2.0 / (W-1), 2.0 / (H-1)]).view(1, 1, 1, 1, 3).to(device=grid.device)
    # B, Ho, Wo, 2
    grid = (grid+1.0)/reso
    x000 = torch.floor(grid).long()

    input = F.pad(input, (0, 1, 0, 1, 0, 1), mode='replicate')

    # B, 8*P: Q000, Q001, Q010, Q011, Q100, Q101, Q110, Q111
    index = torch.stack([(x000[...,1]+x000[...,2]*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+1+x000[...,2]*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+1+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0],
                         (x000[...,1]+x000[...,2]*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+1+x000[...,2]*(H+1))*(W+1)+x000[...,0]+1,
                         (x000[...,1]+1+(x000[...,2]+1)*(H+1))*(W+1)+x000[...,0]+1,
                         ], dim=1).view(B, -1)
    # 2 of (B, C, 4P) corresponding f000, f001, f010, f011, f100, f101, f110, f111
    f0xx, f1xx = torch.gather(input.view(B, C, -1), -1, index.unsqueeze(1).expand(-1, C, -1)).split(P*4, dim=-1)

    # B, Ho, Wo, 3 -> B, P
    xd, yd, zd = torch.unbind((grid - x000).view(B, -1, 3), dim=-1)

    # B, C, 4P
    fxx = f0xx * (1-xd).repeat(1, 4).unsqueeze(1) + f1xx * xd.repeat(1,4).unsqueeze(1)

    # f00, f01: B, C, 2P
    f0x, f1x = fxx.split(2*P, dim=-1)
    fx = f0x * (1-yd).repeat(1, 2).unsqueeze(1) + f1x*yd.repeat(1,2).unsqueeze(1)

    f0, f1 = fx.split(P, dim=-1)
    result = f0*(1-zd).unsqueeze(1) + f1*zd.unsqueeze(1)

    result = result.view(B,C,Do,Ho,Wo)
    # assert(torch.allclose(ref, result))
    return result


if __name__ == "__main__":
    from torch.autograd.gradcheck import gradgradcheck, gradcheck
    torch.manual_seed(0)

    # X = torch.linspace(0, 5, steps=6, dtype=torch.double).view(1, 1, 1, 6).expand(1, 1, 6, -1).contiguous()
    X = torch.rand((1, 1, 6, 6), dtype=torch.double)
    X.requires_grad_()

    grid = torch.rand((1, 3, 3, 2), dtype=torch.double) - 0.5
    grid *= 2.0

    grid.requires_grad_()

    S = grid_sample_2d(X, grid).sum()
    S.backward()
    print(grid.grad)
    grid.grad.zero_()

    S = grid_sample_2d(X, grid).sum()
    grid_grad = autograd.grad(S, grid, create_graph=True)[0]
    S_grad = grid_grad.sum()

    S_grad.backward()
    print(grid.grad)

    # 3D
    X = torch.rand((1, 1, 6, 6, 6), dtype=torch.double)
    X.requires_grad_()
    grid = torch.rand((1, 2, 2, 2, 3), dtype=torch.double) - 0.5
    grid *= 2.0

    grid.requires_grad_()

    S = grid_sample_3d(X, grid).sum()
    S.backward()
    print(grid.grad)
    grid.grad.zero_()

    S = grid_sample_3d(X, grid).sum()
    grid_grad = autograd.grad(S, grid, create_graph=True)[0]
    S_grad = grid_grad.sum()

    S_grad.backward()
    print(grid.grad)
