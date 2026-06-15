#!/usr/bin/env python3
"""Thin entry point -- the converter now lives in `intellifold.convert`.

Kept so the documented `python convert_ifv2_to_jax.py --schema ... --v2-pt ...`
still works. `intellifold predict` auto-downloads + converts weights for you
(see `intellifold.weights`), so you rarely need to run this by hand.
"""
from intellifold.convert import main

if __name__ == "__main__":
    main()
