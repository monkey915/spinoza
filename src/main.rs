use clap::Parser;
use spinoza::physics::state::{BallState, Vec3};
use spinoza::simulation::{simulate, SimError};
use spinoza::table::Table;

/// Simulate a table tennis ball trajectory.
///
/// Coordinate system:
///   Origin = server-side left corner of table surface.
///   +Y = toward opponent, +X = to the right, +Z = up.
///   The table surface is at z = 0.76 m.
///
/// The ball is launched from a position above the server's end of the table.
/// Default launch position: (0.7625, 0.0, 0.90)  (centre-width, end of table, 14 cm above surface).
#[derive(Parser, Debug)]
#[command(
    name = "spinoza",
    version,
    about = "Table tennis ball flight & bounce simulator",
    long_about = None
)]
struct Args {
    /// Launch speed (m/s)
    #[arg(short = 'v', long, default_value_t = 8.0)]
    speed: f64,

    /// Elevation angle above horizontal (degrees, positive = upward)
    #[arg(short = 'e', long, default_value_t = 10.0)]
    elevation: f64,

    /// Azimuth angle from +Y axis (degrees, 0 = straight ahead, positive = right)
    #[arg(short = 'a', long, default_value_t = 0.0)]
    azimuth: f64,

    /// Topspin: angular velocity around -X axis (rad/s, positive = topspin forward)
    ///
    /// Topspin makes the ball dip faster; backspin (negative) floats longer.
    #[arg(long, default_value_t = 0.0)]
    topspin: f64,

    /// Sidespin: angular velocity around +Z axis (rad/s, positive = clockwise from above)
    #[arg(long, default_value_t = 0.0)]
    sidespin: f64,

    /// Backspin: angular velocity around +X axis (rad/s, positive = backspin backward)
    /// (Alias: negative topspin. Mutually exclusive with topspin in practice.)
    #[arg(long, default_value_t = 0.0)]
    backspin: f64,

    /// Launch X position (m), default = table centre width
    #[arg(long, default_value_t = 0.7625)]
    x0: f64,

    /// Launch Y position (m), default = server end (y=0)
    #[arg(long, default_value_t = 0.0)]
    y0: f64,

    /// Launch Z position (m), default = 0.90 m (14 cm above surface)
    #[arg(long, default_value_t = 0.90)]
    z0: f64,

    /// Print full trajectory as CSV
    #[arg(long, default_value_t = false)]
    trajectory: bool,
}

fn main() {
    let args = Args::parse();

    let elev_rad = args.elevation.to_radians();
    let azim_rad = args.azimuth.to_radians();

    // Velocity vector: forward (+Y), height (+Z), sideways (+X)
    let vx = args.speed * elev_rad.cos() * azim_rad.sin();
    let vy = args.speed * elev_rad.cos() * azim_rad.cos();
    let vz = args.speed * elev_rad.sin();

    // Spin: topspin rotates ball around -X (right-hand: ω_x < 0 for topspin)
    // omega_x: topspin = negative (forward roll), backspin = positive
    // omega_z: sidespin
    let omega_x = -args.topspin + args.backspin;
    let omega_y = 0.0_f64;
    let omega_z = -args.sidespin; // clockwise from above = -Z

    let initial = BallState::new(
        Vec3::new(args.x0, args.y0, args.z0),
        Vec3::new(vx, vy, vz),
        Vec3::new(omega_x, omega_y, omega_z),
    );

    let table = Table::standard();

    match simulate(initial, &table) {
        Ok(result) => {
            println!("=== Table Tennis Ball Simulation ===");
            println!();
            println!("Launch position: x={:.4} m  y={:.4} m  z={:.4} m",
                args.x0, args.y0, args.z0);
            println!("Launch velocity: vx={:.3} m/s  vy={:.3} m/s  vz={:.3} m/s",
                vx, vy, vz);
            println!("Spin (ω):        ωx={:.1} rad/s  ωy={:.1} rad/s  ωz={:.1} rad/s",
                omega_x, omega_y, omega_z);
            println!();
            println!("--- Impact after {:.4} s ---", result.bounce_time);
            println!("Impact point:    x={:.4} m  y={:.4} m  (z={:.4} m)",
                result.landing.x, result.landing.y, result.landing.z);
            println!("Vel. before impact:    vx={:.3}  vy={:.3}  vz={:.3}  |v|={:.3} m/s",
                result.pre_bounce.vel.x, result.pre_bounce.vel.y, result.pre_bounce.vel.z,
                result.pre_bounce.vel.norm());
            println!("Vel. after impact:     vx={:.3}  vy={:.3}  vz={:.3}  |v|={:.3} m/s",
                result.post_bounce.vel.x, result.post_bounce.vel.y, result.post_bounce.vel.z,
                result.post_bounce.vel.norm());
            println!("Spin after impact:     ωx={:.1}  ωy={:.1}  ωz={:.1} rad/s",
                result.post_bounce.omega.x, result.post_bounce.omega.y, result.post_bounce.omega.z);

            // Table position info
            println!();
            let on_own_half = result.landing.y < table.length / 2.0;
            let half_label = if on_own_half { "own half" } else { "opponent's half" };
            println!("Table half:      {} (y={:.4} m, centre at y={:.3} m)",
                half_label, result.landing.y, table.length / 2.0);
            println!("Offset from centre: Δx={:.4} m  Δy={:.4} m",
                result.landing.x - table.width / 2.0,
                result.landing.y - table.length / 2.0);

            if args.trajectory {
                println!();
                println!("--- Trajectory ---");
                println!("t_s,x_m,y_m,z_m,vx_ms,vy_ms,vz_ms,ox_rads,oy_rads,oz_rads");
                for (t, s) in &result.trajectory {
                    println!("{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.3},{:.3},{:.3}",
                        t, s.pos.x, s.pos.y, s.pos.z,
                        s.vel.x, s.vel.y, s.vel.z,
                        s.omega.x, s.omega.y, s.omega.z);
                }
            }
        }
        Err(SimError::MissedTable(msg)) => {
            eprintln!("Error: Ball missed the table.");
            eprintln!("  {}", msg);
            std::process::exit(1);
        }
        Err(SimError::HitFloor) => {
            eprintln!("Error: Ball hit the floor before reaching the table.");
            std::process::exit(1);
        }
        Err(SimError::Timeout) => {
            eprintln!("Error: Time limit exceeded (ball in flight too long).");
            std::process::exit(1);
        }
    }
}
