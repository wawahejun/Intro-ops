# MetaX Backend

The training runtime keeps MetaX interfaces aligned with the compiled C ABI
backends and builds MetaX as a separate variant through MACA/cu-bridge.

Current expectations:

- discover the MACA SDK from `MACA_PATH` or `/opt/maca`
- build with `cmake_maca` and `ninja_maca`
- compile `ops/*/metax/*.maca` into `libcamp_ops.so`
- load `_metax` ABI symbols from Python FFI
- validate on a MetaX-enabled PyTorch environment that exposes `cuda:0`
