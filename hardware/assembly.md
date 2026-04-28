# Assembly Guide — Spinoza 4-DOF Table Tennis Arm

This guide adapts the [SO-101 follower arm](https://huggingface.co/docs/lerobot/so101) for table tennis use. We use 4-DOF (instead of 5-DOF + gripper) and replace the gripper with a paddle holder.

## Overview

```
Joint 6 (Gripper)        ← REMOVED
Joint 5 (Wrist Roll)     ← REMOVED
Joint 4 (Wrist Flex)     → phi4: paddle tilt
Joint 3 (Elbow Flex)     → phi3: elbow bend
Joint 2 (Shoulder Lift)  → phi2: shoulder pitch
Joint 1 (Shoulder Pan)   → phi1: shoulder yaw / base rotation
```

The original SO-101 base rotates the entire arm — exactly what we need for phi1.
Joint 2 sits directly above Joint 1 and provides the shoulder pitch motion (phi2),
just like a typical articulated robot.

## Step-by-Step

### Step 1: Print Parts

Follow [`printing.md`](printing.md) to print:
- All SO-ARM101 STL files for joints 1–4
- `paddle_holder.scad` (export to STL via OpenSCAD)

Skip wrist roll and gripper STLs entirely.

### Step 2: Configure Servos

Each servo needs a unique ID on the bus. Use the LeRobot tool **or** our own tool.

#### Option A: LeRobot tool (recommended)

```bash
pip install -e ".[feetech]"
lerobot-find-port  # find your USB port
lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyUSB0
```

This will prompt you to connect each motor one at a time and assign IDs 1–6.
For our 4-DOF version, **only do servos 1–4** and ignore the wrist roll / gripper steps.

#### Option B: Custom tool (in spinoza repo)

```bash
python -c "from robot.servos import FeetechBus; bus = FeetechBus(); bus.open(); print(bus.ping_all())"
```

You'll need to manually flash IDs using a Feetech configuration tool first.

### Step 3: Mechanical Assembly

Follow the [SO-101 official assembly video / guide](https://huggingface.co/docs/lerobot/so101).

**Differences for 4-DOF Spinoza:**

1. **Skip the wrist roll motor** — instead, attach the wrist flex motor (joint 4) directly to the paddle holder.
2. **No gripper** — the paddle holder replaces it entirely.
3. **No leader arm** — we don't need teleoperation.

### Step 4: Mount Paddle Holder

The wrist flex motor (joint 4) has a 4-arm horn on its output shaft.

1. Print `paddle_holder_bottom.stl` (default render of `paddle_holder.scad`)
2. Press 4× M3 heat-set inserts into the boss mounting holes
3. Press 2× M3 heat-set inserts into the clamp end-holes (for clamp screws)
4. Bolt the bottom half to the wrist servo horn (4× M3 screws)
5. Print `paddle_holder_top.stl` (uncomment the second `translate()` line in the SCAD file)
6. Insert paddle handle, tighten 2× M3 screws to clamp

### Step 5: Wire Daisy-Chain

```
PSU 5V --> Waveshare board --> Servo 1 --> Servo 2 --> Servo 3 --> Servo 4
                              (base)    (shoulder)   (elbow)    (wrist)
```

Use the 3-pin TTL servo cables. Each servo has IN and OUT ports — chain them.

### Step 6: Mount to Table

Use 2× G-clamps to bolt the SO-101 base plate to the table.

**Position**: Behind the table on the receiver side, centered on the table width.

For our simulation coordinate system, the arm base should be at:
```
x = 0.7625 m  (table center horizontally)
y = 2.74 m    (far end of table = receiver baseline)
z = 0.76 m    (table surface level)
```

### Step 7: Calibrate Zero Positions

1. Power on with torque disabled:
   ```python
   from robot.arm import RobotArm
   arm = RobotArm()
   arm.connect()
   arm.disable_torque()
   ```

2. Manually move each joint to its **simulation zero pose**:
   - phi1 = 0: arm pointing toward server (-Y direction)
   - phi2 = 0: upper arm horizontal
   - phi3 = 0: forearm straight (no bend)
   - phi4 = 0: paddle face perpendicular to table

3. Read raw positions:
   ```python
   print(arm.bus.read_all_positions())
   ```

4. Copy these values into `robot/config.py` as `ZERO_OFFSETS_RAW`.

### Step 8: Test Movement

```bash
python bridge.py --test-arm
```

This will sweep the arm through a few test positions to verify all joints work correctly.

## Safety Notes

- **Always have an emergency stop**: the simplest is an inline NC switch between PSU and the Waveshare board. Hit it = arm goes limp.
- **Start with low servo speed**: `MOVE_SPEED = 200` in `robot/config.py` for first tests.
- **Keep clear of the workspace** during test runs — even a small motion at 5kg-cm can hurt fingers.
- **Don't run with backlash damage**: if any joint feels loose or sloppy, retighten or reprint.

## Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| Servo doesn't respond | Wrong baudrate / ID | Re-run `lerobot-setup-motors` |
| Arm shakes / oscillates | Wrong P gain or backlash | Reduce `MOVE_SPEED`, tighten joints |
| Servo overheats | Holding torque too high | Reduce `TORQUE_LIMIT` in config |
| IK returns "unreachable" | Target outside reach | Check arm base position config |
| Paddle wobbles | Holder not tight | Tighten clamp screws, add rubber pad |
| One joint's motion is reversed | Servo direction inverted | Negate that joint in `ZERO_OFFSETS_RAW` |
