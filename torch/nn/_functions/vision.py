import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch._thnn import type2backend
from .thnn.auto import function_by_name
import torch.backends.cudnn as cudnn


def affine_grid_generator(theta, size):
    if theta.data.is_cuda and cudnn.enabled and cudnn.is_acceptable(theta.data) and len(size) == 4:
        N, C, H, W = size
        return torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        return AffineGridGenerator.apply(theta, size)


# TODO: Port these completely into C++

class SpatialSliceExtractorTrilinear(Function):

    @staticmethod
    def forward(ctx, input, grid):
        ctx.save_for_backward(input, grid)
        grid_sz = grid.size()
        backend = type2backend[type(input)]
        output = input.new(grid_sz[0], input.size(1), grid_sz[1], grid_sz[2])
        backend.SpatialSliceExtractorTrilinear_updateOutput(backend.library_state, input, grid, output)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        backend = type2backend[type(input)]
        grad_input = input.new(input.size())
        grad_grid = grid.new(grid.size())
        backend.SpatialSliceExtractorTrilinear_updateGradInput(
            backend.library_state, input, grad_input,
            grid, grad_grid, grad_output)
        return grad_input, grad_grid


class AffineGridGenerator(Function):
    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size

        ctx.size = size
        ctx.is_cuda = theta.is_cuda

        if len(size) == 5:
            N, C, D, H, W = size
            base_grid = theta.new(N, D, H, W, 4)

            base_grid[:, :, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, :, 2] = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 3] = 1

            grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)

        elif len(size) == 4:
            N, C, H, W = size
            base_grid = theta.new(N, H, W, 3)

            base_grid[:, :, :, 0] = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            base_grid[:, :, :, 1] = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, 2] = 1

            grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
            grid = grid.view(N, H, W, 2)
        else:
            raise RuntimeError("AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.")

        ctx.base_grid = base_grid

        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        assert ctx.is_cuda == grad_grid.is_cuda
        base_grid = ctx.base_grid

        if len(ctx.size) == 5:
            N, C, D, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, D, H, W, 3])
            grad_theta = torch.bmm(
                base_grid.view(N, D * H * W, 4).transpose(1, 2),
                grad_grid.view(N, D * H * W, 3))
        elif len(ctx.size) == 4:
            N, C, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, H, W, 2])
            grad_theta = torch.bmm(
                base_grid.view(N, H * W, 3).transpose(1, 2),
                grad_grid.view(N, H * W, 2))
        else:
            assert False

        grad_theta = grad_theta.transpose(1, 2)

        return grad_theta, None
