/// 3D vector with basic arithmetic
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[allow(dead_code)]
impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Vec3 { x, y, z } }

    pub fn dot(self, rhs: Vec3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    pub fn cross(self, rhs: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n < 1e-12 { Vec3::ZERO } else { self * (1.0 / n) }
    }

    pub fn scale(self, s: f64) -> Vec3 { self * s }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
}
impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, rhs: Vec3) -> Vec3 { Vec3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z) }
}
impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 { Vec3::new(self.x * s, self.y * s, self.z * s) }
}
impl std::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 { Vec3::new(-self.x, -self.y, -self.z) }
}

/// Full ball state: position, velocity, angular velocity (spin)
#[derive(Debug, Clone, Copy)]
pub struct BallState {
    pub pos: Vec3,    // m
    pub vel: Vec3,    // m/s
    pub omega: Vec3,  // rad/s  (right-hand rule: topspin = -x for ball moving in +y)
}

impl BallState {
    pub fn new(pos: Vec3, vel: Vec3, omega: Vec3) -> Self {
        BallState { pos, vel, omega }
    }
}
