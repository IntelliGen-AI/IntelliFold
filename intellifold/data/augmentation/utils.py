# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang

# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro
#
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.types import Device


def randomly_rotate(
    coords,
    return_second_coords=False,
    second_coords=None,
    generator: Optional[torch.Generator] = None,
):
    R = random_rotations(len(coords), coords.dtype, coords.device, generator=generator)

    if return_second_coords:
        return torch.einsum("bmd,bds->bms", coords, R), (
            torch.einsum("bmd,bds->bms", second_coords, R)
            if second_coords is not None
            else None
        )

    return torch.einsum("bmd,bds->bms", coords, R)


def compute_random_augmentation(
    multiplicity, s_trans=1.0, device=None, dtype=torch.float32
):
    R = random_rotations(multiplicity, dtype=dtype, device=device)
    random_trans = (
        torch.randn((multiplicity, 1, 3), dtype=dtype, device=device) * s_trans
    )
    return R, random_trans


def center_random_augmentation(
    atom_coords,
    atom_mask,
    s_trans=1.0,
    augmentation=True,
    centering=True,
    return_second_coords=False,
    second_coords=None,
    generator: Optional[torch.Generator] = None,
):
    """Center and randomly augment the input coordinates.

    Parameters
    ----------
    atom_coords : Tensor
        The atom coordinates.
    atom_mask : Tensor
        The atom mask.
    s_trans : float, optional
        The translation factor, by default 1.0
    augmentation : bool, optional
        Whether to add rotational and translational augmentation the input, by default True
    centering : bool, optional
        Whether to center the input, by default True
    generator : torch.Generator, optional
        Optional torch.Generator for reproducible rotation/translation.
        When None, falls back to the default global torch RNG.

    Returns
    -------
    Tensor
        The augmented atom coordinates.

    """
    if centering:
        atom_mean = torch.sum(
            atom_coords * atom_mask[:, :, None], dim=1, keepdim=True
        ) / torch.sum(atom_mask[:, :, None], dim=1, keepdim=True)
        atom_coords = atom_coords - atom_mean

        if second_coords is not None:
            # apply same transformation also to this input
            second_coords = second_coords - atom_mean

    if augmentation:
        atom_coords, second_coords = randomly_rotate(
            atom_coords,
            return_second_coords=True,
            second_coords=second_coords,
            generator=generator,
        )
        # Use empty_like().normal_(generator=...) instead of torch.randn_like
        # because the latter does not accept a generator argument.
        random_trans = torch.empty_like(atom_coords[:, 0:1, :]).normal_(generator=generator) * s_trans
        atom_coords = atom_coords + random_trans

        if second_coords is not None:
            second_coords = second_coords + random_trans

    if return_second_coords:
        return atom_coords, second_coords

    return atom_coords


# the following is copied from Torch3D, BSD License, Copyright (c) Meta Platforms, Inc. and affiliates.

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def random_quaternions(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.
        generator: Optional torch.Generator. If given, randn samples are
            drawn from this generator (so the result is reproducible).

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device, generator=generator)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device, generator=generator)
    return quaternion_to_matrix(quaternions)