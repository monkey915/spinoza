// paddle_holder.scad — Table tennis paddle mount for SO-ARM101 wrist servo
//
// Replaces the SO-101 gripper. Bolts onto the SO-ARM101 wrist servo horn
// (4-arm cross horn, M3 mounting pattern) and holds a standard ITTF table
// tennis paddle by clamping the handle.
//
// Coordinate system: Z is the servo rotation axis (up out of the wrist).
// The paddle handle is held along the X axis, with the disc face perpendicular
// to Y (paddle normal = +Y in local frame).
//
// Print: PETG, 0.2mm layers, 35% gyroid infill, 4 perimeters.
// Heat-set inserts: 4x M3 in mounting boss, 2x M3 in clamp halves.

// =========================================================================
// Parameters — adjust to match your hardware
// =========================================================================

// Servo horn pattern (Feetech STS3215 default 4-arm horn)
horn_screw_radius   = 8.0;    // distance from horn center to mounting screw hole
horn_screw_count    = 4;      // 4-arm cross
horn_hub_diameter   = 22.0;   // diameter of the horn hub (boss recess)
horn_hub_height     = 3.0;    // depth of the hub recess

// Paddle handle dimensions (ITTF standard)
handle_length       = 100.0;  // length we want to clamp
handle_width        = 25.0;   // typical handle width
handle_thickness    = 23.0;   // typical handle thickness
clamp_padding       = 1.0;    // extra clearance for tolerance
clamp_grip_thickness = 6.0;   // wall thickness of clamp halves
clamp_screw_offset  = 7.0;    // offset of clamp screws from handle edge

// Mounting boss (mounts to servo horn)
boss_diameter       = 30.0;
boss_height         = 8.0;

// Connector arm (boss -> clamp)
arm_length          = 35.0;   // distance from servo axis to start of clamp
arm_width           = 22.0;
arm_thickness       = 10.0;

// Holes
m3_clearance        = 3.4;    // clearance hole for M3 screws
m3_insert_diameter  = 4.2;    // for heat-set insert (push-fit)
m3_insert_depth     = 5.5;

// Quality
$fn = 64;

// =========================================================================
// Modules
// =========================================================================

module servo_horn_mounting() {
    // Hub recess (so we sit flat on the horn)
    translate([0, 0, -0.01])
        cylinder(h = horn_hub_height + 0.02, d = horn_hub_diameter);

    // Mounting screw clearance holes
    for (i = [0:horn_screw_count-1]) {
        rotate([0, 0, i * 360 / horn_screw_count])
            translate([horn_screw_radius, 0, -1])
                cylinder(h = boss_height + 2, d = m3_clearance);
    }

    // Counter-bores for screw heads (so screw heads sit flush at top)
    for (i = [0:horn_screw_count-1]) {
        rotate([0, 0, i * 360 / horn_screw_count])
            translate([horn_screw_radius, 0, boss_height - 3])
                cylinder(h = 5, d = 6.2);
    }
}

module mounting_boss() {
    difference() {
        union() {
            // Cylindrical boss
            cylinder(h = boss_height, d = boss_diameter);

            // Connecting arm extending in +X to the clamp
            translate([0, -arm_width / 2, 0])
                cube([boss_diameter / 2 + arm_length, arm_width, arm_thickness]);
        }
        servo_horn_mounting();
    }
}

module clamp_half(top = false) {
    // Clamp body that holds the paddle handle
    clamp_total_w = handle_thickness + 2 * clamp_grip_thickness + 2 * clamp_padding;
    clamp_total_l = handle_length + 2 * clamp_grip_thickness;
    clamp_total_h = handle_width / 2 + clamp_grip_thickness + clamp_padding;

    difference() {
        // Outer body
        cube([clamp_total_l, clamp_total_w, clamp_total_h]);

        // Handle pocket (cut out the slot for the handle)
        translate([clamp_grip_thickness,
                   clamp_grip_thickness + clamp_padding,
                   clamp_grip_thickness + clamp_padding])
            cube([handle_length,
                  handle_thickness,
                  handle_width / 2 + 1]);

        // Through-holes for clamp screws (2 screws spanning both halves)
        for (x = [clamp_grip_thickness + clamp_screw_offset,
                  clamp_total_l - clamp_grip_thickness - clamp_screw_offset]) {
            translate([x, -1, clamp_total_h / 2])
                rotate([-90, 0, 0])
                    cylinder(h = clamp_total_w + 2,
                             d = top ? m3_clearance : m3_insert_diameter);
        }

        // For the bottom half: heat-set insert pocket at one end
        if (!top) {
            for (x = [clamp_grip_thickness + clamp_screw_offset,
                      clamp_total_l - clamp_grip_thickness - clamp_screw_offset]) {
                translate([x, clamp_total_w - m3_insert_depth, clamp_total_h / 2])
                    rotate([-90, 0, 0])
                        cylinder(h = m3_insert_depth + 0.5,
                                 d = m3_insert_diameter);
            }
        }
    }
}

// =========================================================================
// Assembly
// =========================================================================

// Bottom half: mounting boss + lower clamp half (single piece)
module paddle_holder_bottom() {
    mounting_boss();

    // Position the clamp half so the paddle handle starts at the end of the arm
    clamp_total_l = handle_length + 2 * clamp_grip_thickness;
    clamp_total_w = handle_thickness + 2 * clamp_grip_thickness + 2 * clamp_padding;

    translate([boss_diameter / 2 + arm_length - 5,
               -clamp_total_w / 2,
               0])
        clamp_half(top = false);
}

// Top half: just the upper clamp body (separate print)
module paddle_holder_top() {
    clamp_total_l = handle_length + 2 * clamp_grip_thickness;
    clamp_total_w = handle_thickness + 2 * clamp_grip_thickness + 2 * clamp_padding;

    translate([0, 0, 0])
        clamp_half(top = true);
}

// =========================================================================
// Render — uncomment one at a time to export each part
// =========================================================================

paddle_holder_bottom();

// translate([0, 60, 0]) paddle_holder_top();
