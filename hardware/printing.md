# 3D Printing Guide — Spinoza Robot Arm

We reuse the SO-ARM101 Follower STL files for joints 1–4 and add a custom paddle holder.

## STL Files

### From the SO-ARM101 Repository

Clone or download from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100):

```bash
git clone https://github.com/TheRobotStudio/SO-ARM100.git
cd SO-ARM100/Meshes/SO101
```

You need these files for a **4-DOF table tennis arm**:

| Joint | STL Files | Notes |
|-------|-----------|-------|
| Base | `Base_SO101.stl`, `Base_Motor_Holder_SO101.stl` | Holds phi1 servo (shoulder yaw) |
| Shoulder | `Shoulder_Pan_SO101.stl`, `Shoulder_Lift_SO101.stl` | Holds phi2 servo (shoulder pitch) |
| Upper arm | `Upper_Arm_SO101.stl` | Connects shoulder to elbow |
| Elbow | `Elbow_SO101.stl` | Holds phi3 servo |
| Forearm | `Forearm_SO101.stl` | Connects elbow to wrist |
| Wrist | `Wrist_Flex_SO101.stl` | Holds phi4 servo |
| **Skip these** | `Wrist_Roll_*.stl`, `Gripper_*.stl` | We use 4-DOF, not 5-DOF + gripper |

### Custom Spinoza Files (in `hardware/`)

| File | Purpose | Replaces |
|------|---------|----------|
| `paddle_holder.scad` | Mounts a real table tennis paddle to the wrist servo horn | The SO-101 gripper |

To generate STL: open in [OpenSCAD](https://openscad.org), then `File -> Export -> STL`.

## Print Settings

### General

| Setting | Value |
|---------|-------|
| Material | PLA (PETG recommended for paddle holder) |
| Layer height | 0.20 mm |
| Wall count | 4 perimeters |
| Top/bottom layers | 5 |
| Infill | 30% gyroid |
| Print speed | 50 mm/s (60 for outer perimeter) |
| Support | Tree supports where needed (mostly for `Base_SO101.stl`) |
| Brim | 4mm (helps adhesion for tall parts) |

### Per-Part Notes

| Part | Orientation | Supports | Notes |
|------|------------|----------|-------|
| `Base_SO101.stl` | Flat side down | Yes | Largest part (~120g) |
| `Shoulder_Pan_SO101.stl` | Servo socket up | No | |
| `Shoulder_Lift_SO101.stl` | Servo socket up | Yes | |
| `Upper_Arm_SO101.stl` | Lay flat | No | |
| `Elbow_SO101.stl` | Servo socket up | Yes | |
| `Forearm_SO101.stl` | Lay flat | No | |
| `Wrist_Flex_SO101.stl` | Servo socket up | Yes | |
| `paddle_holder.stl` | Flat side down | Tree | **Use PETG** for vibration resistance |

### Heat-Set Inserts

After printing parts that need M3 inserts (most of them), use a soldering iron at ~220 C
to press the brass inserts into the printed sockets. Push slowly and straight to avoid
melting around the rim.

The SO-ARM101 STL files have insert sockets pre-modeled — they should accept standard
M3 x 4mm OD x 5mm L heat-set inserts.

## Total Print Time / Material

| Item | Estimate |
|------|----------|
| PLA filament | ~350g |
| PETG (paddle holder) | ~50g |
| Print time at 0.2mm | ~22 hours total |

Print in sequence over 2–3 days; you don't need to babysit each part.

## If You Don't Own a 3D Printer

Order printed parts:
- **EU**: [Craftcloud](https://craftcloud3d.com), [JLC3DP](https://jlc3dp.com)
- Cost estimate: ~€60-90 for the full set in PLA

Or buy a kit (see [BOM.md](BOM.md)) where 3D-printed parts are already included (~€40-60 supplement).
