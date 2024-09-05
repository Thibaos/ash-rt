use std::error::Error;
use std::process::Command;

#[allow(unused)]
macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:warning={}", format!($($tokens)*))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let rt = Command::new("shaders/compile.bat").status()?;
    let ao = Command::new("shaders/AO/compile.bat").status()?;

    if !rt.success() {
        panic!("rt failed {}", rt);
    }

    if !ao.success() {
        panic!("ao failed {}", ao);
    }

    Ok(())
}
