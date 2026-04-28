# Spinoza Hardware

Hardware build documentation for the spinoza table tennis robot arm.

## Files in this folder

| File | Purpose |
|------|---------|
| [`BOM.md`](BOM.md) | Bill of Materials — full shopping list with EU prices and links |
| [`printing.md`](printing.md) | 3D printing guide — STL files, settings, time estimates |
| [`paddle_holder.scad`](paddle_holder.scad) | OpenSCAD source for the paddle clamp (replaces the SO-101 gripper) |
| [`assembly.md`](assembly.md) | Step-by-step assembly + calibration instructions |

## Build Strategy

We do not design our own arm from scratch. Instead, we adapt the proven open-source
[SO-ARM101 Follower](https://github.com/TheRobotStudio/SO-ARM100) by Hugging Face / TheRobotStudio:

- **Same servos** as our software stack (Feetech STS3215)
- **Active community** with assembly videos, troubleshooting, and CAD files
- **~€155 total** for all parts, plus 3D printing
- **Open source** STL files under permissive license

We modify the design in two ways:
1. Use 4-DOF instead of 5-DOF (skip wrist roll motor)
2. Replace the gripper with a custom paddle holder ([`paddle_holder.scad`](paddle_holder.scad))

## Quick Order Path

1. Read [BOM.md](BOM.md), order parts (~10 days from Alibaba for servos)
2. While waiting, print STL files (see [printing.md](printing.md))
3. When parts arrive, follow [assembly.md](assembly.md)
4. Calibrate zero positions (see Step 7 in [assembly.md](assembly.md))
5. Run `python bridge.py --test-arm` to verify

## Total Estimated Cost (EU)

| Component | Cost |
|-----------|------|
| Electronics + servos | ~€87 |
| Screws + inserts + clamps | ~€21 |
| Filament | ~€18 |
| **Total minimum** | **~€126** |
| With E-stop, alu plate, tools | ~€177 |
