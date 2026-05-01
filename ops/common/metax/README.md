# MetaX Backend Stub

The training runtime keeps MetaX interfaces aligned with NVIDIA interfaces, but
does not build MetaX code by default.

A production integration should add:

- MACA SDK discovery (`MACA_PATH`)
- htcc/mxcc compiler rules
- `.maca` source handling
- hcruntime/mcruntime link rules
- backend-specific CI on MetaX hardware

