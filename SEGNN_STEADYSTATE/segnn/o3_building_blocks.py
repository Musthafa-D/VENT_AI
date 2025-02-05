import torch
import torch.nn as nn

from e3nn.o3 import Irreps, Linear, spherical_harmonics, FullyConnectedTensorProduct
from e3nn.nn import Gate

from math import sqrt

class O3TensorProduct(nn.Module):
    """ A bilinear layer, computing CG tensorproduct and normalising them.

    Parameters
    ----------
    irreps_in1 : o3.Irreps
        Input irreps.
    irreps_out : o3.Irreps
        Output irreps.
    irreps_in2 : o3.Irreps
        Second input irreps.
    tp_rescale : bool
        If true, rescales the tensor product.

    """

    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, tp_rescale=True) -> None:
        super().__init__()

        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2
        self.tp_rescale = tp_rescale

        # Build the layers
        self.tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_in1,
            irreps_in2=self.irreps_in2,
            irreps_out=self.irreps_out, shared_weights=True, normalization='component')

        # For each zeroth order output irrep we need a bias
        # So first determine the order for each output tensor and their dims

        # it creates a list in which every element represents an irreps' order
        # the nth element stores the order of the nth irrep
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]

        # it creates a list in which every element is the number of irreps for every order
        # the nth element stores the multiplicity of the nth irrep
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]

        # it creates a list of slices corresponding to indices for each irrep
        # in this way you know how to split the big list with all irreps, when you move from one to the other 
        self.irreps_out_slices = irreps_out.slices()
        # Store tuples of slices and corresponding biases in a list
        self.biases = []
        self.biases_slices = []
        self.biases_slice_idx = []

        # for every level 0 irreps order, it adds a learnable bias (initially set to 0)
        for slice_idx in range(len(self.irreps_out_orders)):
            if self.irreps_out_orders[slice_idx] == 0:
                out_slice = irreps_out.slices()[slice_idx]
                out_bias = torch.zeros(self.irreps_out_dims[slice_idx], dtype=self.tp.weight.dtype)
                self.biases += [out_bias]
                self.biases_slices += [out_slice]
                self.biases_slice_idx += [slice_idx]

        # Initialize the correction factors
        self.slices_sqrt_k = {}

        # Initialize similar to the torch.nn.Linear
        self.tensor_product_init()
        # Adapt parameters so they can be applied using vector operations.
        self.vectorise()

    def tensor_product_init(self) -> None:
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice

            # This loop iterates over pairs of weight (weight tensors) and instr (instructions), to init a FullyConnectedTensorProduct
            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                # Retrieves the slice index from the instruction, to know the connection mode (how multiplicities are treated)
                slice_idx = instr[2]

                # Decomposes the shape of the weight tensor into three variables
                mul_1, mul_2, mul_out = weight.shape
                # reflects the number of input connections or units that contribute to this specific weight (irrep)
                fan_in = mul_1 * mul_2
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)

            for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in

                slice_idx = instr[2]
                # Inside this loop, it calculates the weight initialization scaling factor sqrt_k based on the fan-in.
                # This scaling factor is used to initialize the weights within the specified range
                if self.tp_rescale:
                    sqrt_k = 1 / sqrt(slices_fan_in[slice_idx])
                else:
                    sqrt_k = 1.
                weight.data.uniform_(-sqrt_k, sqrt_k)

                # store the scaling factor for each irreducible representation
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            for (out_slice_idx, out_slice, out_bias) in zip(self.biases_slice_idx, self.biases_slices, self.biases):
                sqrt_k = 1 / sqrt(slices_fan_in[out_slice_idx])
                out_bias.uniform_(-sqrt_k, sqrt_k)

    def vectorise(self):
        """ Adapts the bias parameter and the sqrt_k corrections so they can be applied using vectorised operations """

        # Vectorise the bias parameters
        if len(self.biases) > 0:
            with torch.no_grad():
                # all biases are put in a single vector (Bias parameters are typically associated with zeroth-order irreducible representations (order 0))
                self.biases = torch.cat(self.biases, dim=0)
            self.biases = nn.Parameter(self.biases)

            # Compute broadcast indices.
            # The broadcast indices will be used to apply the bias parameters to the correct elements of the output tensor.
            # This step involves iterating through the irreducible representations and determining which elements correspond to order-0 irreps.
            bias_idx = torch.LongTensor()
            for slice_idx in range(len(self.irreps_out_orders)):
                if self.irreps_out_orders[slice_idx] == 0:
                    out_slice = self.irreps_out.slices()[slice_idx]
                    bias_idx = torch.cat((bias_idx, torch.arange(out_slice.start, out_slice.stop).long()), dim=0)

            # The bias_idx tensor is registered as a buffer using register_buffer. Buffers are persistent tensors that are not considered
            # model parameters but are still stored on the same device as the model's parameters.
            # This buffer will be used to index into the output tensor when applying biases.
            self.register_buffer("bias_idx", bias_idx, persistent=False)
        else:
            self.biases = None

        # Now onto the sqrt_k correction
        # store corrections based on the square root of k values for each irreducible representation
        sqrt_k_correction = torch.zeros(self.irreps_out.dim)
        # iterates through the tensor product instructions (which specify how different irreducible representations are combined)
        for instr in self.tp.instructions:
            # it retrieves the slice_idx, which indicates the corresponding irreducible representation.
            # It also retrieves the sqrt_k value associated with that irreducible representation from the self.slices_sqrt_k dictionary
            slice_idx = instr[2]
            slice, sqrt_k = self.slices_sqrt_k[slice_idx]
            sqrt_k_correction[slice] = sqrt_k

        # Make sure bias_idx and sqrt_k_correction are on same device as module
        self.register_buffer("sqrt_k_correction", sqrt_k_correction, persistent=False)

    def forward_tp_rescale_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 == None:
            # if tensor n.2 is not provided, initializes it as a row of ones with shape [data_in1.shape[0],1]
            data_in2 = torch.ones_like(data_in1[:, 0:1])

        # This line computes the tensor product between data_in1 and data_in2 using the self.tp module or layer
        # (It uses FullyConnectedTensorProduct)
        data_out = self.tp(data_in1, data_in2)

        # Apply corrections
        if self.tp_rescale:
            # this line divides data_out element-wise by self.sqrt_k_correction
            data_out /= self.sqrt_k_correction

        # Add the biases
        if self.biases is not None:
            # If biases are present, this line adds the bias values to the corresponding elements of data_out.
            data_out[:, self.bias_idx] += self.biases
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        return data_out


class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None) -> None:
        # For the gate the output of the linear needs to have an extra number of scalar irreps equal to the amount of
        # non scalar irreps:
        # The first type is assumed to be scalar (Nx0e, so you take N of them) and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated (the resulting number there are as many as irreps)
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # So the gate needs the following irrep as input, this is the output irrep of the tensor product
        # The .simplify() is used to compactify from "Nx0e + Mx0e + [...]" to "(N+M)x0e + [...]"
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2)
        if irreps_g_gated.num_irreps > 0:
            # direct sum of two set of irreps, both passed through different activation functions
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            # if only type-0 irreps are present, simply apply an activation function to them
            self.gate = nn.SiLU()

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_tp_rescale_bias(data_in1, data_in2)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out


# it seems it is not used
class O3SwishGate(torch.nn.Module):
    def __init__(self, irreps_g_scalars, irreps_g_gate, irreps_g_gated) -> None:
        super().__init__()
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in) -> torch.Tensor:
        data_out = self.gate(data_in)
        return data_out