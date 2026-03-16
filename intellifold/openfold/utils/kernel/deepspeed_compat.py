# Copyright 2024 IntelliGen-AI and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import warnings


def detect_deepspeed_status(*, warn_on_import_error: bool = True):
    deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
    if not deepspeed_is_installed:
        return False, False

    try:
        import deepspeed  # noqa: F401
    except Exception as exc:
        if warn_on_import_error:
            warnings.warn(
                "DeepSpeed is installed but could not be imported. "
                "DS4Sci kernels will be unavailable. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        return False, False

    ds4s_is_installed = (
        importlib.util.find_spec("deepspeed.ops.deepspeed4science") is not None
    )
    return True, ds4s_is_installed


def resolve_ds4s_request(enabled: bool):
    if not enabled:
        return False

    _, ds4s_is_installed = detect_deepspeed_status(warn_on_import_error=True)
    if ds4s_is_installed:
        return True

    warnings.warn(
        "USE_DEEPSPEED_EVO_ATTENTION=true but DeepSpeed DS4Sci kernels are "
        "unavailable. Falling back to the standard attention path.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False
