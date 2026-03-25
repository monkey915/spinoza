/// ITTF-standard table tennis table
/// Origin: bottom-left corner of the table surface from the server's side.
/// X-axis: along the width (0 → 1.525 m)
/// Y-axis: along the length (0 → 2.74 m)  — ball travels in +Y direction
/// Z-axis: up; table surface is at z = TABLE_HEIGHT
pub struct Table {
    pub length: f64,     // 2.74 m
    pub width: f64,      // 1.525 m
    pub height: f64,     // 0.76 m (surface height above floor)
    #[allow(dead_code)]
    pub net_height: f64, // 0.1525 m above surface at centre (net clearance check)
}

impl Table {
    pub fn standard() -> Self {
        Table {
            length: 2.74,
            width: 1.525,
            height: 0.76,
            net_height: 0.1525,
        }
    }

    /// Z-coordinate of the table surface
    pub fn surface_z(&self) -> f64 { self.height }

    /// Is (x, y) within the table's horizontal extent?
    pub fn covers_xy(&self, x: f64, y: f64) -> bool {
        x >= 0.0 && x <= self.width && y >= 0.0 && y <= self.length
    }

    /// Compute the fraction t ∈ (0, 1] such that
    ///   pos + t·vel·dt  hits the table surface (z = surface_z)
    /// given a full step of size dt. Returns None if no crossing in [0, dt].
    pub fn time_to_surface(&self, pos: crate::physics::state::Vec3, vel: crate::physics::state::Vec3) -> Option<f64> {
        // Solve: pos.z + t·vel.z = surface_z  →  t = (surface_z - pos.z) / vel.z
        if vel.z >= 0.0 {
            return None; // moving up or horizontal – can't hit table from above
        }
        let t = (self.surface_z() - pos.z) / vel.z;
        if t < 0.0 {
            return None;
        }
        let hit_x = pos.x + vel.x * t;
        let hit_y = pos.y + vel.y * t;
        if self.covers_xy(hit_x, hit_y) {
            Some(t)
        } else {
            None
        }
    }
}
