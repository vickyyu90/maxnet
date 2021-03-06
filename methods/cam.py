import math
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List

__all__ = ['CAM', 'ScoreCAM']


class _CAM:
    """Implements a class activation map extractor

    Args:
        model: input model
        conv_layer: name of the last convolutional layer
    """

    hook_a: Optional[Tensor] = None
    hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def __init__(
        self,
        model: nn.Module,
        conv_layer: str
    ) -> None:

        if not hasattr(model, conv_layer):
            raise ValueError(f"Unable to find submodule {conv_layer} in the model")
        self.model = model
        # Forward hook
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))
        # Enable hooks
        self._hooks_enabled = True
        # Should ReLU be used before normalization
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a = output.data

    def clear_hooks(self) -> None:
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()

    @staticmethod
    def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

        return cams

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> Tensor:

        raise NotImplementedError

    def _precheck(self, class_idx: int, scores: Optional[Tensor] = None) -> None:
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if not isinstance(self.hook_a, Tensor):
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        # Check batch size
        if self.hook_a.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.hook_a.shape[0]}")

        # Check class_idx value
        if class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        #??Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be passed to compute CAMs")

    def __call__(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:

        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM
        return self.compute_cams(class_idx, scores, normalized)

    def compute_cams(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM will be computed
            scores (torch.Tensor[1, K], optional): forward output scores of the hooked model
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        # Get map weight
        weights = self._get_weights(class_idx, scores)
        print("self.hook_a : ", self.hook_a.shape)
        # Perform the weighted combination to get the CAM
        batch_cams = (weights.reshape(weights.shape[0], 1,1, weights.shape[-1]) * self.hook_a.squeeze(0)).sum(0)  # type: ignore[union-attr]

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CAM(_CAM):
    """Implements a class activation map extractor as described in `"Learning Deep Features for Discriminative
    Localization" <https://arxiv.org/pdf/1512.04150.pdf>`_.

    The Class Activation Map (CAM) is defined for image classification models that have global pooling at the end
    of the visual feature extraction block. The localization map is computed as follows:

    .. math::
        L^{(c)}_{CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`,
    and :math:`w_k^{(c)}` is the weight corresponding to class :math:`c` for unit :math:`k` in the fully
    connected layer..

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import CAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = CAM(model, 'layer4', 'fc')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        conv_layer: name of the last convolutional layer
        fc_layer: name of the fully convolutional layer
    """

    def __init__(
        self,
        model: nn.Module,
        conv_layer: str,
        fc_layer: str
    ):

        super().__init__(model, conv_layer)
        # Softmax weight
        self._fc_weights = self.model._modules.get(fc_layer).weight.data

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        # Take the FC weights of the target class
        return self._fc_weights[class_idx, :]


class ScoreCAM(_CAM):
    """Implements a class activation map extractor as described in `"Score-CAM:
    Score-Weighted Visual Explanations for Convolutional Neural Networks" <https://arxiv.org/pdf/1910.01279.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Score-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = softmax(Y^{(c)}(M_k) - Y^{(c)}(X_b))

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the last convolutional layer of the model at
    position :math:`(x, y)`, :math:`Y^{(c)}(X)` is the model output score for class :math:`c` before softmax
    for input :math:`X`, :math:`X_b` is a baseline image,
    and :math:`M_k` is defined as follows:

    .. math::
        M_k = \\frac{U(A_k) - \\min\\limits_m U(A_m)}{\\max\\limits_m  U(A_m) - \\min\\limits_m  U(A_m)})
        \\odot X

    where :math:`\\odot` refers to the element-wise multiplication and :math:`U` is the upsampling operation.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        conv_layer: name of the last convolutional layer
        input_layer: name of the first layer
        batch_size: batch size used to forward masked inputs
    """

    def __init__(
        self,
        model: nn.Module,
        conv_layer: str,
        input_layer: str,
        batch_size: int = 32
    ) -> None:

        super().__init__(model, conv_layer)

        # Input hook
        self.hook_handles.append(self.model._modules.get(input_layer).register_forward_pre_hook(self._store_input))
        self.bs = batch_size
        # Ensure ReLU is applied to CAM before normalization
        self._relu = True

    def _store_input(self, module: nn.Module, input: Tensor):
        """Store model input tensor"""

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        # Normalize the activation
        self.hook_a: Tensor
        upsampled_a = self._normalize(self.hook_a)

        #??Upsample it to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(upsampled_a, self._input.shape[-3:], mode='trilinear', align_corners=False)

        # Use it as a mask
        # O * I * H * W
        masked_input = upsampled_a.squeeze(0).unsqueeze(1) * self._input

        # Initialize weights
        weights = torch.zeros(masked_input.shape[0], dtype=masked_input.dtype).to(device=masked_input.device)

        # Disable hook updates
        self._hooks_enabled = False
        #??Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(weights.shape[0] / self.bs)):

            selection_slice = idx * self.bs
            with torch.no_grad():
                #??Get the softmax probabilities of the target class
                out = self.model(masked_input[selection_slice].unsqueeze(1))
                weights[selection_slice] = F.softmax(out, dim=1)[:, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(batch_size={self.bs})"
