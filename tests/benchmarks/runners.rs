use std::process::Stdio;
use std::process::Output;
use std::io::Read;
use std::time::Duration;
use wait_timeout::ChildExt;
use crate::is_nvcc_installed;
use crate::Config;
use regex::Regex;
use std::io::stderr;
use std::io::Write;
use std::process::Command;

fn run_compiler<'a, I>(extra_args: I) -> Result<Output, std::io::Error>
where
    I: IntoIterator<Item = String>,
{
    Command::new("cargo")
        .args(["run", "--"])
        .args(extra_args)
        .output()
}

fn run_make(dir: &str, variant: Option<&str>) -> Result<bool, std::io::Error> {
    if is_nvcc_installed() {
        if variant.is_some() {
            Command::new("make")
                .current_dir(dir)
                .arg(variant.unwrap())
                .output()
                .map(|o| o.status.success())
        } else {
            Command::new("make")
                .current_dir(dir)
                .output()
                .map(|o| o.status.success())
        }
    } else {
        eprintln!("Skipping make!");
        Ok(true)
    }
}


fn run_bin(bin: &str, input_file: &str, timeout: Duration) -> Result<String, String> {
    if is_nvcc_installed() {
        let child = Command::new(bin)
                                .arg(input_file)
                                .stdout(Stdio::piped())
                                .stderr(Stdio::piped())
                                .spawn();
        if child.is_err() {
            return Err("Binary would not run.".to_string());
        }

        let mut ok_child = child.unwrap();
        let out_status;
        let mut stdout = String::new();
        let mut stderr = String::new();
        match ok_child.wait_timeout(timeout).unwrap() {
            Some(status) => {
                out_status = status;
                
                if let Some(mut out) = ok_child.stdout {
                    out.read_to_string(&mut stdout)
                       .expect("Could not convert to string");
                }

                if let Some(mut out) = ok_child.stderr {
                    out.read_to_string(&mut stderr)
                       .expect("Could not convert to string");
                }
            }
            None => {
                ok_child.kill().unwrap();
                ok_child.wait().unwrap();
                return Err("timeout".to_string())
            }
        };

        if !out_status.success() {
            let reason = if stderr.contains("Could not fit kernel on device!") {
                "kernel did not fit"
            } else if stderr.contains("slot < capacity") {
                "capacity"
            } else {
                "other"
            };
            return Err(format!("non-zero exitcode ({reason})."));
        }
        Ok(stdout)
    } else {
        Err("skipped".to_string())
    }
}

pub fn bench_testcases(
    testcases: &Vec<(&str, Vec<&str>)>,
    bin: &str,
    csv_prefix: &str,
    formatter: fn(&str) -> String,
    reps: usize,
    result_file: &mut dyn Write,
    timeout: Duration
) {
    for (prob_category, files) in testcases {
        eprintln!("\t\t\tTesting {prob_category}...");
        for f in files {
            let res = bench_file(bin, f, reps, timeout);
            result_file
                .write_all(
                    format!("{csv_prefix},{prob_category},{},{res}\n", formatter(f)).as_bytes(),
                )
                .expect("Could not write result of testcase.");
        }
    }
}

pub fn compile_config(
    pname: &str,
    dir: &str,
    make_variant: Option<&str>,
    config: &Config,
    extra_args: Vec<String>,
) -> Result<(), String> {
    adl_compile(pname, dir, config, extra_args)?;
    cuda_compile(dir, make_variant)?;

    Ok(())
}

fn get_runtime_ms_from_stdout(output: &str) -> Option<f32> {
    let re = Regex::new(r"Total walltime GPU: ([0-9]+\.[0-9]+) ms").unwrap();
    for l in output.lines() {
        let ms = re
            .captures(l)
            .map(|c| c.get(1).unwrap().as_str().parse::<f32>().unwrap());
        if ms.is_some() {
            return ms;
        }
    }
    return None;
}

fn adl_compile(
    pname: &str,
    dir: &str,
    config: &Config,
    mut extra_args: Vec<String>,
) -> Result<(), String> {
    eprintln!("\t\tCompiling ADL code...");
    let in_file = format!("{dir}/{pname}.adl");
    let out_file = format!("{dir}/{pname}.cu");

    let mut args: Vec<String> = vec![in_file, "-o".to_string(), out_file, "-t".to_string()];
    args.append(&mut config.to_args());
    args.append(&mut extra_args);
    let r = run_compiler(args);

    if r.is_err() {
        Err("\t\t\tFailed to start ADL compiler.".to_string())
    } else if !r.as_ref().unwrap().status.success() {
        let stderr = String::from_utf8_lossy(&r.unwrap().stderr).to_string();
        Err(format!("\t\t\tFailed to compile ADL code: {}", stderr))
    } else {
        Ok(())
    }
}

fn cuda_compile(dir: &str, make_variant: Option<&str>) -> Result<(), String> {
    eprintln!("\t\tCompiling CUDA code...");
    let r = run_make(dir, make_variant);

    if r.is_err() {
        Err("\t\t\tFailed to start Make.".to_string())
    } else if !r.unwrap() {
        Err("\t\t\tFailed to compile CUDA code.".to_string())
    } else {
        Ok(())
    }
}

fn bench_file<'a>(bin: &str, f: &str, reps: usize, timeout: Duration) -> String {
    let mut total_ms = 0.0;

    for i in 0..reps {
        eprint!("\r\t\t\t\tStarting input {f} ({}/{reps})", i + 1);
        stderr().flush().unwrap();
        let r = run_bin(bin, f, timeout);

        if r.is_err() {
            let e = r.unwrap_err();
            eprintln!("\n\t\t\t\t\t{e}");
            return e;
        } else {
            let stdout = r.unwrap();
            let ms = get_runtime_ms_from_stdout(&stdout);
            if ms.is_none() {
                eprintln!("\n\t\t\t\t\tCould not find time information in output.");
                return "NaN".to_string();
            }

            total_ms += ms.unwrap();
        }
    }
    eprint!("\n");

    return (total_ms / (reps as f32)).to_string();
}
