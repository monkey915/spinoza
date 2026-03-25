pub mod physics;
pub mod table;
pub mod simulation;
pub mod rally;
pub mod serve;

// PyO3 Python bindings (only compiled with `python` feature)
#[cfg(feature = "python")]
mod pymodule;

#[cfg(feature = "python")]
pub use pymodule::spinoza as python_module;
