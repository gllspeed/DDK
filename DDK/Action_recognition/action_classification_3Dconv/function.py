from torch.autograd import Function


class FakeQuantize(Function):

    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    ##########STE：直通估计器，反量化后的输出对权重求梯度，然后更新量化之前的权重。
    ####由于卷积中用的 weight 是经过伪量化操作的，因此可以模拟量化误差，
    # 把这些误差的梯度回传到原来的 weight，又可以更新权重，使其适应量化产生的误差
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None