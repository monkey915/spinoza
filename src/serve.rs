use crate::physics::state::{BallState, Vec3};

/// Curriculum difficulty level for serve generation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Difficulty {
    /// Slow serves, no spin — learn basic contact
    Stage1,
    /// Medium speed, light topspin/backspin
    Stage2,
    /// Full range: fast serves, heavy spin, sidespin
    Stage3,
}

/// Parameters defining the random range for serves at each difficulty
struct ServeRange {
    speed: (f64, f64),
    elevation: (f64, f64),
    azimuth: (f64, f64),
    topspin: (f64, f64),
    backspin: (f64, f64),
    sidespin: (f64, f64),
}

impl Difficulty {
    fn range(self) -> ServeRange {
        match self {
            Difficulty::Stage1 => ServeRange {
                speed: (5.0, 7.0),
                elevation: (-25.0, -10.0), // downward: aimed at server's half
                azimuth: (-2.0, 2.0),
                topspin: (0.0, 0.0),
                backspin: (0.0, 0.0),
                sidespin: (0.0, 0.0),
            },
            Difficulty::Stage2 => ServeRange {
                speed: (5.0, 10.0),
                elevation: (-30.0, -8.0),
                azimuth: (-5.0, 5.0),
                topspin: (0.0, 100.0),
                backspin: (0.0, 60.0),
                sidespin: (-30.0, 30.0),
            },
            Difficulty::Stage3 => ServeRange {
                speed: (4.0, 14.0),
                elevation: (-35.0, -5.0),
                azimuth: (-15.0, 15.0),
                topspin: (0.0, 200.0),
                backspin: (0.0, 150.0),
                sidespin: (-150.0, 150.0),
            },
        }
    }
}

/// Simple xorshift64 PRNG (no external dependency needed)
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f64 in [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [lo, hi]
    fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

/// Generate a random serve at the given difficulty level.
///
/// Realistic serve model: ball is struck from above (~1.0-1.1m) with a slight
/// downward angle, aimed to bounce on the server's half first, then arc over
/// the net to the receiver's side.
pub fn random_serve(rng: &mut Rng, difficulty: Difficulty) -> BallState {
    let r = difficulty.range();

    let speed = rng.uniform_range(r.speed.0, r.speed.1);
    // Downward elevation: ball is hit from above, aimed at server's half
    let elevation_deg = rng.uniform_range(r.elevation.0, r.elevation.1);
    let azimuth_deg = rng.uniform_range(r.azimuth.0, r.azimuth.1);

    let elev_rad = elevation_deg.to_radians();
    let azim_rad = azimuth_deg.to_radians();

    let vx = speed * elev_rad.cos() * azim_rad.sin();
    let vy = speed * elev_rad.cos() * azim_rad.cos();
    let vz = speed * elev_rad.sin(); // negative = downward

    // Generate spin — only one of topspin/backspin active per serve
    let topspin = rng.uniform_range(r.topspin.0, r.topspin.1);
    let backspin = rng.uniform_range(r.backspin.0, r.backspin.1);
    let sidespin = rng.uniform_range(r.sidespin.0, r.sidespin.1);

    // Pick topspin OR backspin (not both)
    let (omega_x, spin_type) = if rng.uniform() < 0.5 {
        (-topspin, "topspin")
    } else {
        (backspin, "backspin")
    };
    let _ = spin_type;
    let omega_z = -sidespin;

    // Launch from slightly behind the server's end, at racket height (~1.0-1.1m)
    let launch_z = rng.uniform_range(1.00, 1.10);
    let pos = Vec3::new(0.7625, 0.10, launch_z);

    BallState::new(pos, Vec3::new(vx, vy, vz), Vec3::new(omega_x, 0.0, omega_z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serves_are_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        let s1 = random_serve(&mut rng1, Difficulty::Stage2);
        let s2 = random_serve(&mut rng2, Difficulty::Stage2);
        assert_eq!(s1.vel.x, s2.vel.x);
        assert_eq!(s1.vel.y, s2.vel.y);
        assert_eq!(s1.omega.x, s2.omega.x);
    }

    #[test]
    fn test_stage1_no_spin() {
        let mut rng = Rng::new(123);
        for _ in 0..100 {
            let s = random_serve(&mut rng, Difficulty::Stage1);
            assert_eq!(s.omega.x, 0.0);
            assert_eq!(s.omega.y, 0.0);
            assert_eq!(s.omega.z, -0.0); // sidespin range is (0,0)
        }
    }

    #[test]
    fn test_serves_vary() {
        let mut rng = Rng::new(999);
        let s1 = random_serve(&mut rng, Difficulty::Stage3);
        let s2 = random_serve(&mut rng, Difficulty::Stage3);
        // Very unlikely to generate identical serves
        assert!(
            s1.vel.x != s2.vel.x || s1.vel.y != s2.vel.y,
            "Consecutive serves should differ"
        );
    }
}
