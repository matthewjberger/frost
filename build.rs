fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    if target_os == "windows" && target_env == "msvc" {
        println!("cargo:rustc-link-arg-bins=/STACK:134217728");
    } else if target_os == "windows" && target_env == "gnu" {
        println!("cargo:rustc-link-arg-bins=-Wl,--stack,134217728");
    } else if target_os == "linux" {
        println!("cargo:rustc-link-arg-bins=-Wl,-z,stack-size=134217728");
    } else if target_os == "macos" {
        println!("cargo:rustc-link-arg-bins=-Wl,-stack_size,0x8000000");
    }
}
