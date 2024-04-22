use std::error::Error;
use std::process::Command;

fn main() -> Result<(), Box<dyn Error>> {
    let status = Command::new("shaders/compile.bat").status()?;

    if !status.success() {
        panic!("failed to execute process: {status}")
    }

    Ok(())
}
