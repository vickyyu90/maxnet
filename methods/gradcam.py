import torch
from torch import Tensor
from typing import Optional

from .cam import _CAM

__all__ = ['GradCAM', 'GradCAMpp']


class _GradCAM(_CAM):
    """Implements a gradient-based class activation map extractor

    Args:
        model: input model
        conv_layer: name of the last convolutional layer
    """

    hook_g: Optional[Tensor] = None

    def __init__(
        self,
        model: torch.nn.Module,
        conv_layer: str
    ) -> None:

        super().__init__(model, conv_layer)
        # Ensure ReLU is applied before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = True
        # Backward hook
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))

    def _hook_g(self, module: torch.nn.Module, input: Tensor, output: Tensor) -> None:
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_g = output[0].data

    def _backprop(self, scores: Tensor, class_idx: int) -> None:
        """Backpropagate the loss for a specific output class"""

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Backpropagate to get the gradients on the hooked layer
        loss = scores[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, class_idx, scores):

        raise NotImplementedError


class GradCAM(_GradCAM):
    """Implements a class activation map extractor as described in `"Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" <https://arxiv.org/pdf/1610.02391.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{H \\cdot W} \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W
        \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import GradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAM(model, 'layer4')
        >>> with torch.no_grad(): scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        conv_layer: name of the last convolutional layer
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)
        # Global average pool the gradients over spatial dimensions
        return self.hook_g.squeeze(0).mean(dim=(1, 2))


class GradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in `"Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks" <https://arxiv.org/pdf/1710.11063.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W \\alpha_k^{(c)}(i, j) \\cdot
        ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax,
    and :math:`\\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \\alpha_k^{(c)}(i, j) = \\frac{1}{\\sum\\limits_{i, j} \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}}
        = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot
        \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} + \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}}

    if :math:`\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1` else :math:`0`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import GradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAMpp(model, 'layer4')
        >>> with torch.no_grad(): scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        conv_layer: name of the last convolutional layer
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)
        #??Alpha coefficient for each pixel
        grad_2 = self.hook_g.pow(2)
        grad_3 = grad_2 * self.hook_g
        alpha = grad_2 / (2 * grad_2 + (grad_3 * self.hook_a).sum(dim=(2, 3), keepdim=True))

        #??Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(dim=(1, 2))
