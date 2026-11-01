from dataclasses import dataclass
from typing import Callable, List, Tuple, Union, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.cait import Cait
from scipy.stats import special_ortho_group
import warnings

from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

from moviad.models.fastflow.loss_functions import fastflow_loss


@dataclass
class FastFlowTrainArgs(TrainingArgs):

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters()
            )
        if self.loss_function is None:
            self.loss_function = fastflow_loss


def create_fastflow(img_shape, backbone_name, device):
    fast_flow_model = CompleteFastFlowModel(backbone_name, input_size=img_shape, normalize=True, device=device)
    fast_flow_module = FastflowModel(
        input_size=img_shape, flow_steps=8, conv3x3_only=False,
        hidden_ratio=1.0, channels=fast_flow_model.channels, scales=fast_flow_model.scales
    )
    fast_flow_model.fast_flow_module = fast_flow_module
    fast_flow_model.to(device)
    return fast_flow_model

class InvertibleModule(nn.Module):
    r"""Base class for all invertible modules in FrEIA.

    Given ``module``, an instance of some InvertibleModule.
    This ``module`` shall be invertible in its input dimensions,
    so that the input can be recovered by applying the module
    in backwards mode (``rev=True``), not to be confused with
    ``pytorch.backward()`` which computes the gradient of an operation::
        x = torch.randn(BATCH_SIZE, DIM_COUNT)
        c = torch.randn(BATCH_SIZE, CONDITION_DIM)
        # Forward mode
        z, jac = module([x], [c], jac=True)
        # Backward mode
        x_rev, jac_rev = module(z, [c], rev=True)
    The ``module`` returns :math:`\\log \\det J = \\log \\left| \\det \\frac{\\partial f}{\\partial x} \\right|`
    of the operation in forward mode, and
    :math:`-\\log | \\det J | = \\log \\left| \\det \\frac{\\partial f^{-1}}{\\partial z} \\right| = -\\log \\left| \\det \\frac{\\partial f}{\\partial x} \\right|`
    in backward mode (``rev=True``).
    Then, ``torch.allclose(x, x_rev) == True`` and ``torch.allclose(jac, -jac_rev) == True``.
    """

    def __init__(self, dims_in: Iterable[Tuple[int]], dims_c: Iterable[Tuple[int]] = None):
        """Initialize.

        Args:
            dims_in: list of tuples specifying the shape of the inputs to this
                     operator: ``dims_in = [shape_x_0, shape_x_1, ...]``
            dims_c:  list of tuples specifying the shape of the conditions to
                     this operator.
        """
        super().__init__()
        if dims_c is None:
            dims_c = []
        self.dims_in = list(dims_in)
        self.dims_c = list(dims_c)

    def forward(
        self, x_or_z: Iterable[Tensor], c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True
    ) -> Tuple[Tuple[Tensor], Tensor]:
        r"""Forward/Backward Pass.

        Perform a forward (default, ``rev=False``) or backward pass (``rev=True``) through this module/operator.

        **Note to implementers:**
        - Subclasses MUST return a Jacobian when ``jac=True``, but CAN return a
          valid Jacobian when ``jac=False`` (not punished). The latter is only recommended
          if the computation of the Jacobian is trivial.
        - Subclasses MUST follow the convention that the returned Jacobian be
          consistent with the evaluation direction. Let's make this more precise:
          Let :math:`f` be the function that the subclass represents. Then:
          .. math::
              J &= \\log \\det \\frac{\\partial f}{\\partial x} \\\\
              -J &= \\log \\det \\frac{\\partial f^{-1}}{\\partial z}.
          Any subclass MUST return :math:`J` for forward evaluation (``rev=False``),
          and :math:`-J` for backward evaluation (``rev=True``).

        Args:
            x_or_z: input data (array-like of one or more tensors)
            c:      conditioning data (array-like of none or more tensors)
            rev:    perform backward pass
            jac:    return Jacobian associated to the direction
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not provide forward(...) method")

    def log_jacobian(self, *args, **kwargs):
        """This method is deprecated, and does nothing except raise a warning."""
        raise DeprecationWarning(
            "module.log_jacobian(...) is deprecated. "
            "module.forward(..., jac=True) returns a "
            "tuple (out, jacobian) now."
        )

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        """Use for shape inference during construction of the graph.

        MUST be implemented for each subclass of ``InvertibleModule``.

        Args:
          input_dims: A list with one entry for each input to the module.
            Even if the module only has one input, must be a list with one
            entry. Each entry is a tuple giving the shape of that input,
            excluding the batch dimension. For example for a module with one
            input, which receives a 32x32 pixel RGB image, ``input_dims`` would
            be ``[(3, 32, 32)]``

        Returns:
            A list structured in the same way as ``input_dims``. Each entry
            represents one output of the module, and the entry is a tuple giving
            the shape of that output. For example if the module splits the image
            into a right and a left half, the return value should be
            ``[(3, 16, 32), (3, 16, 32)]``. It is up to the implementor of the
            subclass to ensure that the total number of elements in all inputs
            and all outputs is consistent.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not provide output_dims(...)")

class AllInOneBlock(InvertibleModule):
    r"""Module combining the most common operations in a normalizing flow or similar model.

    It combines affine coupling, permutation, and global affine transformation
    ('ActNorm'). It can also be used as GIN coupling block, perform learned
    householder permutations, and use an inverted pre-permutation. The affine
    transformation includes a soft clamping mechanism, first used in Real-NVP.
    The block as a whole performs the following computation:
    .. math::
        y = V\\,R \\; \\Psi(s_\\mathrm{global}) \\odot \\mathrm{Coupling}\\Big(R^{-1} V^{-1} x\\Big)+ t_\\mathrm{global}
    - The inverse pre-permutation of x (i.e. :math:`R^{-1} V^{-1}`) is optional (see
      ``reverse_permutation`` below).
    - The learned householder reflection matrix
      :math:`V` is also optional all together (see ``learned_householder_permutation``
      below).
    - For the coupling, the input is split into :math:`x_1, x_2` along
      the channel dimension. Then the output of the coupling operation is the
      two halves :math:`u = \\mathrm{concat}(u_1, u_2)`.
      .. math::
          u_1 &= x_1 \\odot \\exp \\Big( \\alpha \\; \\mathrm{tanh}\\big( s(x_2) \\big)\\Big) + t(x_2) \\\\
          u_2 &= x_2
      Because :math:`\\mathrm{tanh}(s) \\in [-1, 1]`, this clamping mechanism prevents
      exploding values in the exponential. The hyperparameter :math:`\\alpha` can be adjusted.
    """

    def __init__(
        self,
        dims_in,
        dims_c=[],
        subnet_constructor: Callable = None,
        affine_clamping: float = 2.0,
        gin_block: bool = False,
        global_affine_init: float = 1.0,
        global_affine_type: str = "SOFTPLUS",
        permute_soft: bool = False,
        learned_householder_permutation: int = 0,
        reverse_permutation: bool = False,
    ):
        r"""Initialize.

        Args:
            dims_in (_type_): dims_in
            dims_c (list, optional): dims_c. Defaults to [].
            subnet_constructor (Callable, optional): class or callable ``f``, called as ``f(channels_in, channels_out)`` and
                should return a torch.nn.Module. Predicts coupling coefficients :math:`s, t`. Defaults to None.
            affine_clamping (float, optional): clamp the output of the multiplicative coefficients before
                exponentiation to +/- ``affine_clamping`` (see :math:`\\alpha` above). Defaults to 2.0.
            gin_block (bool, optional): Turn the block into a GIN block from Sorrenson et al, 2019.
                Makes it so that the coupling operations as a whole is volume preserving. Defaults to False.
            global_affine_init (float, optional): Initial value for the global affine scaling :math:`s_\mathrm{global}`.. Defaults to 1.0.
            global_affine_type (str, optional): ``'SIGMOID'``, ``'SOFTPLUS'``, or ``'EXP'``. Defines the activation to be used
                on the beta for the global affine scaling (:math:`\\Psi` above).. Defaults to "SOFTPLUS".
            permute_soft (bool, optional): bool, whether to sample the permutation matrix :math:`R` from :math:`SO(N)`,
                or to use hard permutations instead. Note, ``permute_soft=True`` is very slow
                when working with >512 dimensions. Defaults to False.
            learned_householder_permutation (int, optional): Int, if >0, turn on the matrix :math:`V` above, that represents
                multiple learned householder reflections. Slow if large number.
                Dubious whether it actually helps network performance. Defaults to 0.
            reverse_permutation (bool, optional): Reverse the permutation before the block, as introduced by Putzky
                et al, 2019. Turns on the :math:`R^{-1} V^{-1}` pre-multiplication above. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
        """

        super().__init__(dims_in, dims_c)

        channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))

        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            assert tuple(dims_c[0][1:]) == tuple(
                dims_in[0][1:]
            ), f"Dimensions of input and condition don't agree: {dims_c} vs {dims_in}."
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        split_len1 = channels - channels // 2
        split_len2 = channels // 2
        self.splits = [split_len1, split_len2]

        try:
            self.permute_function = {0: F.linear, 1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[self.input_rank]
        except KeyError:
            raise ValueError(f"Data is {1 + self.input_rank}D. Must be 1D-4D.")

        self.in_channels = channels
        self.clamp = affine_clamping
        self.GIN = gin_block
        self.reverse_pre_permute = reverse_permutation
        self.householder = learned_householder_permutation

        if permute_soft and channels > 512:
            warnings.warn(
                (
                    "Soft permutation will take a very long time to initialize "
                    f"with {channels} feature channels. Consider using hard permutation instead."
                )
            )

        # global_scale is used as the initial value for the global affine scale
        # (pre-activation). It is computed such that
        # global_scale_activation(global_scale) = global_affine_init
        # the 'magic numbers' (specifically for sigmoid) scale the activation to
        # a sensible range.
        if global_affine_type == "SIGMOID":
            global_scale = 2.0 - np.log(10.0 / global_affine_init - 1.0)
            self.global_scale_activation = lambda a: 10 * torch.sigmoid(a - 2.0)
        elif global_affine_type == "SOFTPLUS":
            global_scale = 2.0 * np.log(np.exp(0.5 * 10.0 * global_affine_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.global_scale_activation = lambda a: 0.1 * self.softplus(a)
        elif global_affine_type == "EXP":
            global_scale = np.log(global_affine_init)
            self.global_scale_activation = lambda a: torch.exp(a)
        else:
            raise ValueError('Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"')

        self.global_scale = nn.Parameter(
            torch.ones(1, self.in_channels, *([1] * self.input_rank)) * float(global_scale)
        )
        self.global_offset = nn.Parameter(torch.zeros(1, self.in_channels, *([1] * self.input_rank)))

        if permute_soft:
            w = special_ortho_group.rvs(channels)
        else:
            w = np.zeros((channels, channels))
            for i, j in enumerate(np.random.permutation(channels)):
                w[i, j] = 1.0

        if self.householder:
            # instead of just the permutation matrix w, the learned housholder
            # permutation keeps track of reflection vectors vk, in addition to a
            # random initial permutation w_0.
            self.vk_householder = nn.Parameter(0.2 * torch.randn(self.householder, channels), requires_grad=True)
            self.w_perm = None
            self.w_perm_inv = None
            self.w_0 = nn.Parameter(torch.FloatTensor(w), requires_grad=False)
        else:
            self.w_perm = nn.Parameter(
                torch.FloatTensor(w).view(channels, channels, *([1] * self.input_rank)), requires_grad=False
            )
            self.w_perm_inv = nn.Parameter(
                torch.FloatTensor(w.T).view(channels, channels, *([1] * self.input_rank)), requires_grad=False
            )

        if subnet_constructor is None:
            raise ValueError("Please supply a callable subnet_constructor" "function or object (see docstring)")
        self.subnet = subnet_constructor(self.splits[0] + self.condition_channels, 2 * self.splits[1])
        self.last_jac = None

    def _construct_householder_permutation(self):
        """Compute a permutation matrix.

        Compute a permutation matrix from the reflection vectors that are
        learned internally as nn.Parameters.
        """
        w = self.w_0
        for vk in self.vk_householder:
            w = torch.mm(w, torch.eye(self.in_channels).to(w.device) - 2 * torch.ger(vk, vk) / torch.dot(vk, vk))

        for i in range(self.input_rank):
            w = w.unsqueeze(-1)
        return w

    def _permute(self, x, rev=False):
        """Perform permutation.

        Performs the permutation and scaling after the coupling operation.
        Returns transformed outputs and the LogJacDet of the scaling operation.
        """
        if self.GIN:
            scale = 1.0
            perm_log_jac = 0.0
        else:
            scale = self.global_scale_activation(self.global_scale)
            perm_log_jac = torch.sum(torch.log(scale))

        if rev:
            return ((self.permute_function(x, self.w_perm_inv) - self.global_offset) / scale, perm_log_jac)
        else:
            return (self.permute_function(x * scale + self.global_offset, self.w_perm), perm_log_jac)

    def _pre_permute(self, x, rev=False):
        """Permute before the coupling block, only used if reverse_permutation is set."""
        if rev:
            return self.permute_function(x, self.w_perm)
        else:
            return self.permute_function(x, self.w_perm_inv)

    def _affine(self, x, a, rev=False):
        """Perform affine coupling operation.

        Given the passive half, and the pre-activation outputs of the
        coupling subnetwork, perform the affine coupling operation.
        Returns both the transformed inputs and the LogJacDet.
        """

        # the entire coupling coefficient tensor is scaled down by a
        # factor of ten for stability and easier initialization.
        a *= 0.1
        ch = x.shape[1]

        sub_jac = self.clamp * torch.tanh(a[:, :ch])
        if self.GIN:
            sub_jac -= torch.mean(sub_jac, dim=self.sum_dims, keepdim=True)

        if not rev:
            return (x * torch.exp(sub_jac) + a[:, ch:], torch.sum(sub_jac, dim=self.sum_dims))
        else:
            return ((x - a[:, ch:]) * torch.exp(-sub_jac), -torch.sum(sub_jac, dim=self.sum_dims))

    def forward(self, x, c=[], rev=False, jac=True):
        """See base class docstring."""
        if self.householder:
            self.w_perm = self._construct_householder_permutation()
            if rev or self.reverse_pre_permute:
                self.w_perm_inv = self.w_perm.transpose(0, 1).contiguous()

        if rev:
            x, global_scaling_jac = self._permute(x[0], rev=True)
            x = (x,)
        elif self.reverse_pre_permute:
            x = (self._pre_permute(x[0], rev=False),)

        x1, x2 = torch.split(x[0], self.splits, dim=1)

        if self.conditional:
            x1c = torch.cat([x1, *c], 1)
        else:
            x1c = x1

        if not rev:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1)
        else:
            a1 = self.subnet(x1c)
            x2, j2 = self._affine(x2, a1, rev=True)

        log_jac_det = j2
        x_out = torch.cat((x1, x2), 1)

        if not rev:
            x_out, global_scaling_jac = self._permute(x_out, rev=False)
        elif self.reverse_pre_permute:
            x_out = self._pre_permute(x_out, rev=True)

        # add the global scaling Jacobian to the total.
        # trick to get the total number of non-channel dimensions:
        # number of elements of the first channel of the first batch member
        n_pixels = x_out[0, :1].numel()
        log_jac_det += (-1) ** rev * n_pixels * global_scaling_jac

        return (x_out,), log_jac_det

    def output_dims(self, input_dims):
        """Output Dims."""
        return input_dims

class SequenceINN(InvertibleModule):
    """Simpler than FrEIA.framework.GraphINN.

    Only supports a sequential series of modules (no splitting, merging,
    branching off).
    Has an append() method, to add new blocks in a more simple way than the
    computation-graph based approach of GraphINN. For example:
    .. code-block:: python
       inn = SequenceINN(channels, dims_H, dims_W)
       for i in range(n_blocks):
           inn.append(FrEIA.modules.AllInOneBlock, clamp=2.0, permute_soft=True)
       inn.append(FrEIA.modules.HaarDownsampling)
       # and so on
    """

    def __init__(self, *dims: int, force_tuple_output=False):
        super().__init__([dims])

        self.shapes = [tuple(dims)]
        self.conditions = []
        self.module_list = nn.ModuleList()

        self.force_tuple_output = force_tuple_output

    def append(self, module_class, cond=None, cond_shape=None, **kwargs):
        """Append a reversible block from FrEIA.modules to the network.

        Args:
          module_class: Class from FrEIA.modules.
          cond (int): index of which condition to use (conditions will be passed as list to forward()).
            Conditioning nodes are not needed for SequenceINN.
          cond_shape (tuple[int]): the shape of the condition tensor.
          **kwargs: Further keyword arguments that are passed to the constructor of module_class (see example).
        """

        dims_in = [self.shapes[-1]]
        self.conditions.append(cond)

        if cond is not None:
            kwargs["dims_c"] = [cond_shape]

        module = module_class(dims_in, **kwargs)
        self.module_list.append(module)
        ouput_dims = module.output_dims(dims_in)
        assert len(ouput_dims) == 1, "Module has more than one output"
        self.shapes.append(ouput_dims[0])

    def __getitem__(self, item):
        """Get item."""
        return self.module_list.__getitem__(item)

    def __len__(self):
        """Get length."""
        return self.module_list.__len__()

    def __iter__(self):
        """Iter."""
        return self.module_list.__iter__()

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        """Output Dims."""
        if not self.force_tuple_output:
            raise ValueError(
                "You can only call output_dims on a SequentialINN " "when setting force_tuple_output=True."
            )
        return input_dims

    def forward(
        self, x_or_z: Tensor, c: Iterable[Tensor] = None, rev: bool = False, jac: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Execute the sequential INN in forward or inverse (rev=True) direction.

        Args:
            x_or_z: input tensor (in contrast to GraphINN, a list of
                    tensors is not supported, as SequenceINN only has
                    one input).
            c: list of conditions.
            rev: whether to compute the network forward or reversed.
            jac: whether to compute the log jacobian
        Returns:
            z_or_x (Tensor): network output.
            jac (Tensor): log-jacobian-determinant.
        """

        iterator = range(len(self.module_list))
        log_det_jac = 0

        if rev:
            iterator = reversed(iterator)

        if torch.is_tensor(x_or_z):
            x_or_z = (x_or_z,)
        for i in iterator:
            if self.conditions[i] is None:
                x_or_z, j = self.module_list[i](x_or_z, jac=jac, rev=rev)
            else:
                x_or_z, j = self.module_list[i](x_or_z, c=[c[self.conditions[i]]], jac=jac, rev=rev)
            log_det_jac = j + log_det_jac

        return x_or_z if self.force_tuple_output else x_or_z[0], log_det_jac


def subnet_conv_func(kernel_size: int, hidden_ratio: float) -> Callable:
    """Subnet Convolutional Function.

    Callable class or function ``f``, called as ``f(channels_in, channels_out)`` and
        should return a torch.nn.Module.
        Predicts coupling coefficients :math:`s, t`.

    Args:
        kernel_size (int): Kernel Size
        hidden_ratio (float): Hidden ratio to compute number of hidden channels.

    Returns:
        Callable: Sequential for the subnet constructor.
    """

    def subnet_conv(in_channels: int, out_channels: int) -> nn.Sequential:
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv

def create_fast_flow_block(
    input_dimensions: List[int],
    conv3x3_only: bool,
    hidden_ratio: float,
    flow_steps: int,
    clamp: float = 2.0,
) -> SequenceINN:
    """Create NF Fast Flow Block.

    This is to create Normalizing Flow (NF) Fast Flow model block based on
    Figure 2 and Section 3.3 in the paper.

    Args:
        input_dimensions (List[int]): Input dimensions (Channel, Height, Width)
        conv3x3_only (bool): Boolean whether to use conv3x3 only or conv3x3 and conv1x1.
        hidden_ratio (float): Ratio for the hidden layer channels.
        flow_steps (int): Flow steps.
        clamp (float, optional): Clamp. Defaults to 2.0.

    Returns:
        SequenceINN: FastFlow Block.
    """
    nodes = SequenceINN(*input_dimensions)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size if isinstance(input_size, tuple) else tuple(input_size)

    def forward(self, hidden_variables: List[Tensor]) -> Tensor:
        """Generate Anomaly Heatmap.

        This implementation generates the heatmap based on the flow maps
        computed from the normalizing flow (NF) FastFlow blocks. Each block
        yields a flow map, which overall is stacked and averaged to an anomaly
        map.

        Args:
            hidden_variables (List[Tensor]): List of hidden variables from each NF FastFlow block.

        Returns:
            Tensor: Anomaly Map.
        """
        flow_maps: List[Tensor] = []
        for hidden_variable in hidden_variables:
            log_prob = -torch.mean(hidden_variable**2, dim=1, keepdim=True) * 0.5
            prob = torch.exp(log_prob)
            flow_map = F.interpolate(
                input=-prob,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )
            flow_maps.append(flow_map)
        flow_maps = torch.stack(flow_maps, dim=-1)
        anomaly_map = torch.mean(flow_maps, dim=-1)

        return anomaly_map


class CompleteFastFlowModel(VADModel):
    def __init__(self, backbone_name, input_size, normalize, device):
        super().__init__()
        self.device = device

        if backbone_name in ["cait_m48_448", "deit_base_distilled_patch16_384"]:
            feature_extractor = timm.create_model(backbone_name, pretrained=True)
        elif backbone_name in ["resnet18", "wide_resnet50_2"]:
            feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )

       
        self.input_size = input_size
        self.feature_extractor = feature_extractor
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)

        if backbone_name in ["cait_m48_448", "deit_base_distilled_patch16_384"]:
            channels = [768]
            scales = [16]
        elif backbone_name in ["resnet18", "wide_resnet50_2"]:
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm
            self.norms = nn.ModuleList()
            for channel, scale in zip(channels, scales):
                if not normalize:
                    self.norms.append( nn.Identity() ) 
                else:
                    self.norms.append(
                        nn.LayerNorm(
                            [channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                            elementwise_affine=True,
                        )
                    )
        else:
            raise ValueError(
                f"Backbone {backbone_name} is not supported. List of available backbones are "
                "[cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2]."
            )

        # freeze the feature extractor
        self.feature_extractor.eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.channels = channels
        self.scales = scales


    def forward(self, input_tensor):
        if isinstance(self.feature_extractor, VisionTransformer):
            features = self._get_vit_features(input_tensor)
        elif isinstance(self.feature_extractor, Cait):
            features = self._get_cait_features(input_tensor)
        else:
            features = self._get_cnn_features(input_tensor)

        hidden_variables, log_jacobians = self.fast_flow_module(features)

        if self.training:
            return hidden_variables, log_jacobians
        else:
            return self.post_process(hidden_variables)

    def post_process(self, hidden_variables: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Generate anomaly maps from hidden variables.

        Args:
            hidden_variables (List[Tensor]): Hidden variables from NF blocks.

        Returns:
            Tuple[Tensor, Tensor]: (anomaly_maps, anomaly_scores)
        """
        anomaly_maps = self.anomaly_map_generator(hidden_variables)
        anomaly_scores = anomaly_maps.view(anomaly_maps.shape[0], -1).max(dim=1).values
        return anomaly_maps, anomaly_scores

    def to(self, device: torch.device):
        super().to(device)
        self.feature_extractor.to(device)
        if hasattr(self, 'norms'):
            self.norms.to(device)
        self.fast_flow_module.to(device)
        self.anomaly_map_generator.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_extractor.eval()
        self.fast_flow_module.train(mode)
        return self

    def parameters(self):
        return self.fast_flow_module.parameters()

    def train_epoch(self, epoch, train_dataloader, training_args: TrainingArgs):
        avg_batch_loss = 0
        for batch in tqdm(train_dataloader):
            avg_batch_loss += self.train_step(batch, training_args)
        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    def compute_loss(self, batch: torch.Tensor, training_args: TrainingArgs) -> torch.Tensor:
        batch = batch.to(self.device)
        hidden_variables, log_jacobians = self.forward(batch)
        return training_args.loss_function(hidden_variables, log_jacobians)

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        loss = self.compute_loss(batch, training_args)

        training_args.optimizer.zero_grad()
        loss.backward()
        training_args.optimizer.step()
        return loss.item()


    def _get_cnn_features(self, input_tensor: Tensor) -> List[Tensor]:
        """Get CNN-based features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            List[Tensor]: List of features.
        """
        with torch.no_grad():
            features = self.feature_extractor(input_tensor.to(self.device)) 
            #features = features.to(self.device_1)
            # for i,feature in enumerate(features):
            #     print(feature.shape)
            #     print(feature.device)
                
            features = [feature for feature in features]
            features = [self.norms[i](feature) for i, feature in enumerate(features)]
            return features

    def _get_cait_features(self, input_tensor: Tensor) -> List[Tensor]:
        """Get Class-Attention-Image-Transformers (CaiT) features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            List[Tensor]: List of features.
        """
        feature = self.feature_extractor.patch_embed(input_tensor)
        feature = feature + self.feature_extractor.pos_embed
        feature = self.feature_extractor.pos_drop(feature)
        for i in range(41):  # paper Table 6. Block Index = 40
            feature = self.feature_extractor.blocks[i](feature)
        batch_size, _, num_channels = feature.shape
        feature = self.feature_extractor.norm(feature)
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        features = [feature]
        return features

    def _get_vit_features(self, input_tensor: Tensor) -> List[Tensor]:
        """Get Vision Transformers (ViT) features.

        Args:
            input_tensor (Tensor): Input Tensor.

        Returns:
            List[Tensor]: List of features.
        """
        # print(f"get_vit_features input_tensor: {input_tensor.shape}")
        feature = self.feature_extractor.patch_embed(input_tensor)
        cls_token = self.feature_extractor.cls_token.expand(feature.shape[0], -1, -1)
        if self.feature_extractor.dist_token is None:
            feature = torch.cat((cls_token, feature), dim=1)
        else:
            feature = torch.cat(
                (
                    cls_token,
                    self.feature_extractor.dist_token.expand(feature.shape[0], -1, -1),
                    feature,
                ),
                dim=1,
            )
        feature = self.feature_extractor.pos_drop(feature + self.feature_extractor.pos_embed)
        for i in range(8):  # paper Table 6. Block Index = 7
            feature = self.feature_extractor.blocks[i](feature)
        feature = self.feature_extractor.norm(feature)
        feature = feature[:, 2:, :]
        batch_size, _, num_channels = feature.shape
        feature = feature.permute(0, 2, 1)
        feature = feature.reshape(batch_size, num_channels, self.input_size[0] // 16, self.input_size[1] // 16)
        features = [feature]
        return features


class FastflowModel(nn.Module):
    
    """FastFlow.

    Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows.

    Args:
        input_size (Tuple[int, int]): Model input size.
        backbone (str): Backbone CNN network
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
        flow_steps (int, optional): Flow steps.
        conv3x3_only (bool, optinoal): Use only conv3x3 in fast_flow model. Defaults to False.
        hidden_ratio (float, optional): Ratio to calculate hidden var channels. Defaults to 1.0.

    Raises:
        ValueError: When the backbone is not supported.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        flow_steps: int = 8,
        conv3x3_only: bool = False,
        hidden_ratio: float = 1.0,
        channels: list[int] = [64, 128, 256],
        scales: list[int] = [1, 2, 4],
    ) -> None:
        super().__init__()

        self.input_size = input_size

        self.fast_flow_blocks = nn.ModuleList()
        
        for channel, scale in zip(channels, scales):
            print(f"channel:{channel} scale: {scale}")
            self.fast_flow_blocks.append(
                create_fast_flow_block(
                    input_dimensions=[channel, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        

    def forward(self, features: Tensor) -> Union[Tuple[List[Tensor], List[Tensor]], Tensor]:
        """Forward-Pass the input to the FastFlow Model.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            Union[Tuple[Tensor, Tensor], Tensor]: During training, return
                (hidden_variables, log-of-the-jacobian-determinants).
                During the validation/test, return the anomaly map.
        """

        return_val: Union[Tuple[List[Tensor], List[Tensor]], Tensor]

        # Compute the hidden variable f: X -> Z and log-likelihood of the jacobian
        # (See Section 3.3 in the paper.)
        # NOTE: output variable has z, and jacobian tuple for each fast-flow blocks.
        hidden_variables: List[Tensor] = []
        log_jacobians: List[Tensor] = []
        for fast_flow_block, feature in zip(self.fast_flow_blocks, features):
            hidden_variable, log_jacobian = fast_flow_block(feature)
            hidden_variables.append(hidden_variable)
            log_jacobians.append(log_jacobian)

        return_val = (hidden_variables, log_jacobians)

        return return_val

    def inverse(self, hidden_variables: List[Tensor]) -> List[Tensor]:
        """Inverse-Pass of the normalizing flows f^-1: Z -> X.

        Args:
            hidden_variables (List[Tensor]): Hidden variables, one for each fast-flow block.

        Returns:
            List[Tensor]: Features (after the normalization layers) mapped to the given hidden variables.
        """
        features: List[Tensor] = []
        for fast_flow_block, hidden_variable in zip(self.fast_flow_blocks, hidden_variables):
            feature, _ = fast_flow_block(hidden_variable, rev=True)
            features.append(feature)

        return features

