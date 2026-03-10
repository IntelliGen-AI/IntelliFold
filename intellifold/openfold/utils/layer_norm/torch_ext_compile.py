# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from torch.utils.cpp_extension import load


def _cuda_arch_flags():
    raw = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not raw:
        return []

    flags = []
    for entry in raw.replace(",", ";").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        wants_ptx = entry.endswith("+PTX")
        entry = entry.removesuffix("+PTX").replace("sm_", "").replace("compute_", "")
        if "." in entry:
            major, minor = entry.split(".", 1)
            arch = f"{major}{minor}"
        else:
            arch = entry
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        if wants_ptx:
            flags.extend(["-gencode", f"arch=compute_{arch},code=compute_{arch}"])
    return flags


def compile(name, sources, extra_include_paths, build_directory):
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
            "-std=c++17",
            "-maxrregcount=50",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            *_cuda_arch_flags(),
        ],
        verbose=True,
        build_directory=build_directory,
    )
