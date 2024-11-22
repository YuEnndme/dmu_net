import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)                       # 用于保存输入和目标 以便在反向传播过程 backward() 函数使用这些保存的变量计算梯度。
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))     # 计算预测结果和真实标签的交集
        self.union = torch.sum(input) + torch.sum(target) + eps     # 计算预测结果和真实标签的并集

        t = (2 * self.inter.float() + eps) / self.union.float()     # 计算 Dice 系数公式中的分子部分
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    # 计算每对输入和目标样本的Dice系数
    '''
    Dice系数是一种衡量两个样本之间相似度的统计指标，通常用于评估图像分割模型的性能。它计算预测的二值化掩码与真实标签之间的相似程度。
    Dice系数的计算公式为：Dice=(2×intersection+ε)/(union+ε)
    其中，ε 是一个很小的值，用于防止分母为零。intersection 表示预测掩码和真实标签的交集，union 表示它们的并集。Dice系数的取值范围为0到1，1表示完美匹配，0表示完全不匹配。
    input表示mask_pred预测掩码    target表示true_mask真实掩码
    '''
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)
