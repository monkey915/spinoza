# Bill of Materials — Spinoza Robot Arm

Based on the [SO-ARM101 Follower](https://github.com/TheRobotStudio/SO-ARM100) design,
adapted for table tennis (4-DOF + paddle holder instead of 5-DOF + gripper).

## Total cost: ~€155 (EU) excluding shipping and 3D printing

---

## Electronics & Servos

| # | Part | Qty | Unit Price (EU) | Total | Source |
|---|------|-----|----------------|-------|--------|
| 1 | **Feetech STS3215 Servo** 7.4V, 1/345 gear (C001) | 4 | €12.20 | €48.80 | [Alibaba](https://www.alibaba.com/product-detail/Top-Seller-Low-Cost-Feetech-STS3215_1600999461525.html) |
| 2 | **Waveshare Bus Servo Adapter** (USB → TTL) | 1 | €11.40 | €11.40 | [Amazon DE](https://www.amazon.de/dp/B0CJ6TP3TP) |
| 3 | **USB-C Cable** 1m | 1 | €4.00 | €4.00 | Any source |
| 4 | **Power Supply** 5V/5A or 7.4V/3A barrel jack | 1 | €15.70 | €15.70 | [Amazon DE](https://www.amazon.de/dp/B01HRR9GY4) |
| 5 | **3-pin TTL servo cables** 200mm (for daisy-chain) | 5 | €1.50 | €7.50 | Comes with motors usually |

### ⚡ Speed vs. Torque — Gear Ratio Choice

**This is the most important decision in the BOM.** The STS3215 comes in multiple gear
ratios with very different speed/torque tradeoffs:

| Gear ratio | Max speed (no-load, 7.4V) | Stall torque | 90° move | Good for |
|-----------|---------------------------|--------------|----------|----------|
| **1/345** (C001) | 50 RPM ≈ 300°/s ≈ 5.2 rad/s | 16.5 kg·cm | ~300 ms | High torque, slow defensive returns |
| **1/191** (C046) | 90 RPM ≈ 540°/s ≈ 9.4 rad/s | 9.2 kg·cm | ~170 ms | **Recommended for table tennis** ⭐ |
| **1/147** | 110 RPM ≈ 660°/s ≈ 11.5 rad/s | 7.0 kg·cm | ~135 ms | Wrist only — too weak for shoulder |

**Reality check for table tennis:**
- A human topspin swing: ~8–15 rad/s at the hand
- A pingpong ball needs ~400 ms reaction after our prediction kicks in
- 1/345 → robot can barely return slow balls; no real "swinging" motion
- 1/191 → can reach prediction in time and add some swing momentum
- Pro-grade speed (15+ rad/s) requires Dynamixel XM430 / Robstride at ~€200/servo

**Recommended configurations:**

1. **Cheap-and-slow (€155)**: 4× 1/345 — first prototype, prove the pipeline works.
   Robot will only do passive returns, no real shots.

2. **⭐ Balanced for pingpong (€155)**: 4× 1/191 — same price, same Waveshare adapter,
   same `robot/servos.py` driver. Order with `gear_ratio=191` from Alibaba.
   You lose ~45% torque but gain ~80% speed. **This is what we'd recommend.**

3. **Mixed build (€155)**: 1/345 for shoulder pitch (joint 2 fights gravity), 1/191
   for the rest. Best torque-where-needed, fastest-where-it-matters compromise.

4. **Pro upgrade (€800+)**: Dynamixel XM430-W350 or Robstride 01 for the two distal
   joints. Driver code would need rewriting. Only worth it if you want offensive shots.

**At 5V instead of 7.4V**, all speeds drop ~30% and torque drops ~25%. Plan for 7.4V
if you care about performance.

> **Note on power**: The 7.4V STS3215 can run at 5V (reduced torque ~12 kg·cm) or 7.4V
> (full torque). Start with 5V — easier to source and safer. Upgrade later if needed.

---

## Mechanical Hardware (Screws, Inserts, etc.)

| # | Part | Qty | Unit Price | Total | Notes |
|---|------|-----|-----------|-------|-------|
| 6 | **M3×6 socket head screws** | 50 | €0.05 | €2.50 | DIN 912 |
| 7 | **M3×8 socket head screws** | 30 | €0.05 | €1.50 | DIN 912 |
| 8 | **M3×12 socket head screws** | 20 | €0.06 | €1.20 | DIN 912 |
| 9 | **M3×16 socket head screws** | 10 | €0.07 | €0.70 | DIN 912 |
| 10 | **M2×6 self-tapping screws** | 30 | €0.05 | €1.50 | For servo horns |
| 11 | **M3 brass heat-set inserts** | 30 | €0.10 | €3.00 | [Amazon DE](https://www.amazon.de/s?k=m3+heat+set+insert) — for press-fit into PLA |
| 12 | **M3 hex nuts** | 20 | €0.03 | €0.60 | DIN 934 |

> All screws available as a mixed set: search "M3 sortiment Innensechskant" on Amazon (~€20 box covers everything above).

---

## Mounting / Base

| # | Part | Qty | Unit Price | Total | Notes |
|---|------|-----|-----------|-------|-------|
| 13 | **Table clamps** (G-clamp 100mm) | 2 | €4.85 | €9.70 | [Amazon DE](https://www.amazon.de/s?k=schraubzwinge+100mm) |
| 14 | **Aluminum base plate** 200×200×6mm | 1 | €15.00 | €15.00 | Optional — alternative is bolt to wood plank |

---

## 3D Printing Materials

| # | Part | Qty | Unit Price | Total | Notes |
|---|------|-----|-----------|-------|-------|
| 15 | **PLA filament** 1kg spool | 1 | €18.00 | €18.00 | Total print uses ~400g — 1 spool is plenty |
| 16 | **PETG filament** 1kg (optional, more durable) | 1 | €22.00 | €22.00 | Recommended for paddle holder |

---

## Tools (One-Time)

| # | Part | Qty | Unit Price | Total | Notes |
|---|------|-----|-----------|-------|-------|
| 17 | **Hex key set** (1.5–6mm) | 1 | €8 | €8 | If you don't have one |
| 18 | **Soldering iron** for heat-set inserts | 1 | €15 | €15 | If you don't have one |

---

## Optional / Recommended

| # | Part | Qty | Unit Price | Total | Notes |
|---|------|-----|-----------|-------|-------|
| 19 | **Emergency-stop button** (NC switch in series with PSU) | 1 | €8 | €8 | **Strongly recommended** — kill power instantly |
| 20 | **Cable sleeve / spiral wrap** | 2m | €5 | €5 | Keep daisy-chain tidy |

---

## Summary by Category

| Category | Cost |
|----------|------|
| Servos & electronics | ~€87 |
| Screws / inserts | ~€11 |
| Mounting | ~€10 |
| Filament | ~€18 |
| **Subtotal — minimum to build** | **~€126** |
| Optional (E-stop, alu plate, sleeves) | ~€28 |
| Tools (if you don't have them) | ~€23 |
| **Maximum total** | **~€177** |

---

## Where to Order Everything (EU)

The fastest single-package path:

1. **Servos**: 4× STS3215 7.4V 1/345 from Alibaba (€55 incl. shipping, ~10 days)
2. **Control board + power**: Waveshare board from [Amazon.de](https://www.amazon.de/dp/B0CJ6TP3TP) (~2 days)
3. **Screws + inserts**: One M3 sortiment box from Amazon (~€20)
4. **Filament**: Whatever your printer uses, or order from [Prusa](https://www.prusa3d.com/category/filaments/)
5. **Clamps**: From Amazon or local hardware store

---

## Pre-Built Kit Alternative (Easier, More Expensive)

If you don't want to source parts individually, complete kits exist:

- **Seeed Studio**: [SO-ARM100 Kit](https://www.seeedstudio.com/SO-ARM100-Low-Cost-AI-Arm-Kit.html) (~$199 + shipping)
- **Autodiscovery (EU)**: [SO-101 Kit](https://autodiscovery.eu/en/products/so-101-kit) — fully assembled or DIY
- **WowRobo**: Pre-assembled version available

These kits include 6 servos (we only need 4) — you'd have 2 spares for upgrades.
