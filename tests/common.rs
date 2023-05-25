use std::process::ExitStatus;
use std::process::Command;
use std::env;

pub fn run_benchmarks() -> bool {
    if let Ok(v) = env::var("BENCHMARK") {
        return v == "true";
    }
    return false;
}

pub fn run_compiler<'a, I>(extra_args : I) -> Result<ExitStatus, std::io::Error>
where
    I: IntoIterator<Item = &'a str>
{
    Command::new("cargo")
        .args(["run", "--"])
        .args(extra_args)
        .output()
        .map(|o| o.status)
}

pub fn run_make(dir : &str)-> Result<ExitStatus, std::io::Error> {
    Command::new("make")
        .current_dir(dir)
        .output()
        .map(|o| o.status)
}