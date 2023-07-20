use std::io::Read;
use std::io::stderr;
use regex::Regex;
use std::process::Stdio;
use core::time::Duration;
use std::process::Output;
use wait_timeout::ChildExt;
use std::io::Write;
use std::fs::File;
use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fmt::Display;
use std::process::Command;

const REPS: usize = 5;
pub type TestCase<'a> = (&'a str, Vec<(&'a str, Vec<String>, usize)>);


pub fn benchmark(tests: &Vec<(&Config, &Vec<TestCase>)>, bin_name: &str, bin_folder: &str) {
    if !is_benchmarking() {
        eprintln!("@@@@@ SKIPPING {bin_name} @@@@@");
        return;
    }

    let mut result_file = File::create(format!("{bin_folder}/{bin_name}_results.csv"))
        .expect("Could not create benchmark csv file.");
    result_file
        .write_all(
            format!(
                "{},algorithm,problem_type,problem_size,runtime\n",
                Config::HEADER
            )
            .as_bytes(),
        )
        .expect("Could not write header.");

    let mut last_extra_args : Option<&Vec<String>> = None;

    eprintln!("\tAlgorithm: {bin_name}");
    for (config_idx, (config, testcases)) in tests.iter().enumerate() {
        eprintln!("\tConfig: {config}({}/{})", config_idx + 1, tests.len());
        
        for (problem_type_idx, (problem_type, files_and_args)) in testcases.iter().enumerate() {
            eprintln!("\tProblem: {problem_type}({}/{})", problem_type_idx + 1, testcases.len());

            for (test_idx, (file, extra_args, p_size)) in files_and_args.iter().enumerate() {
                let file_name = file.split("/").last().unwrap();
                eprintln!("\tFile: {file_name}({}/{})", test_idx + 1, files_and_args.len());
                
                // Compile if needed
                if last_extra_args.is_none() || extra_args != last_extra_args.unwrap() {
                    let compile_err = compile_config(config, bin_name, bin_folder, extra_args).err();
                    if let Some(e) = compile_err {
                        eprintln!("{}", e);
                        continue;
                    }
                    last_extra_args = Some(extra_args);
                }

                
                let csv_prefix = format!("{},{bin_name},{problem_type},{p_size}", config.as_csv_row());
                let runtime = bench_file(&format!("{bin_folder}/{bin_name}.out"), file, REPS, Duration::from_secs(45));
                result_file.write_all(format!("{csv_prefix},{runtime}\n").as_bytes()).expect("Could not write to result file.");
            }
        }
        last_extra_args = None;
    }
}

pub fn memorder_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed", "seqcons", "acqrel"];
    let voting = [
        ("on-host", "on-host-simple"),
        ("in-kernel", "in-kernel-simple"),
        ("graph", "graph-simple"),
    ];
    let w = ["0", "1"];

    Config::cartesian(&orders, &voting, &w)
}

pub fn voting_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting = [
        ("in-kernel", "in-kernel-simple"),
        ("in-kernel", "in-kernel-alternating"),
        ("on-host", "on-host-simple"),
        ("on-host", "on-host-alternating"),
        ("graph", "graph-shared"),
        ("graph", "graph-shared-banks"),
        ("graph", "graph-shared-opportunistic"),
        ("graph", "graph-simple")
    ];
    let w = ["1"];

    Config::cartesian(&orders, &voting, &w)
}

pub fn is_benchmarking() -> bool {
    if let Ok(v) = env::var("BENCHMARK") {
        return v == "true";
    }
    return false;
}

pub fn is_nvcc_installed() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map_or(false, |s| s.status.success())
}


#[derive(Eq, PartialEq, Hash, Clone)]
pub struct Config<'a> {
    pub memorder: &'a str,
    pub schedule: &'a str,
    pub voting: &'a str,
    pub weak_non_racing: &'a str
}

impl<'a> Config<'_> {
    pub const HEADER: &str = "memorder,schedule,voting-strat,weak_non_racing";

    pub fn new(m: &'a str, s: &'a str, v: &'a str, w: &'a str) -> Config<'a> {
        Config {
            memorder: m,
            schedule: s,
            voting: v,
            weak_non_racing: w
        }
    }

    pub fn to_args(&self) -> Vec<String> {
        vec![
            "-m",
            self.memorder,
            "-S",
            self.schedule,
            "-v",
            self.voting,
            "-w",
            self.weak_non_racing,
        ].iter().map(|s| s.to_string()).collect()
    }

    pub fn as_csv_row(&self) -> String {
        format!(
            "{},{},{},{}",
            self.memorder, self.schedule, self.voting, self.weak_non_racing
        )
    }

    pub fn cartesian(
        orders: &[&'static str],
        voting: &[(&'static str, &'static str)],
        weak_non_racing: &[&'static str],
    ) -> Vec<Config<'static>> {
        let mut result: Vec<Config> = Vec::new();

        for o in orders {
            for (s, v) in voting {
                for w in weak_non_racing {
                    result.push(Config::new(o, s, v, w));
                }
            }
        }
        result
    }

    pub fn union(configs: &'a Vec<Vec<Config<'a>>>) -> HashSet<&'a Config<'a>> {
        let flat: Vec<&Config<'a>> = configs.iter().map(|v| v.iter()).flatten().collect();

        HashSet::from_iter(flat)
    }
}

impl Display for Config<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tuple = format!(
            "{}.{}.{}.{}",
            self.memorder, self.schedule, self.voting, self.weak_non_racing
        );
        write!(f, "{}", tuple)
    }
}


fn run_compiler<'a, I>(extra_args: I) -> Result<Output, std::io::Error>
where
    I: IntoIterator<Item = String>,
{
    Command::new("cargo")
        .args(["run", "--"])
        .args(extra_args)
        .output()
}

fn run_make(bin_name: &str, dir: &str) -> Result<bool, std::io::Error> {
    if is_nvcc_installed() {
        Command::new("make")
            .current_dir(dir)
            .arg(bin_name)
            .output()
            .map(|o| o.status.success())
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
            if reason == "other" {
                eprintln!("\nSTDERR: {}", stderr);
            }
            return Err(format!("non-zero exitcode ({reason})."));
        }
        Ok(stdout)
    } else {
        Err("skipped".to_string())
    }
}

fn compile_config(
    config: &Config,
    bin_name: &str,
    dir: &str,
    extra_args: &Vec<String>,
) -> Result<(), String> {
    adl_compile(bin_name, dir, config, extra_args)?;
    cuda_compile(bin_name, dir)?;

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
    bin_name: &str,
    dir: &str,
    config: &Config,
    extra_args: &Vec<String>,
) -> Result<(), String> {
    eprintln!("\tCompiling ADL code...");
    let in_file = format!("{dir}/{bin_name}.adl");
    let out_file = format!("{dir}/{bin_name}.cu");

    let mut args: Vec<String> = vec![in_file, "-o".to_string(), out_file, "-t".to_string()];
    args.append(&mut config.to_args());
    args.append(&mut extra_args.clone());
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

fn cuda_compile(bin_name: &str, dir: &str) -> Result<(), String> {
    eprintln!("\tCompiling CUDA code...");
    let r = run_make(bin_name, dir);

    if r.is_err() {
        Err("\t\t\tFailed to start Make.".to_string())
    } else if !r.unwrap() {
        Err("\t\t\tFailed to compile CUDA code.".to_string())
    } else {
        Ok(())
    }
}

fn bench_file<'a>(bin: &str, input_file: &str, reps: usize, timeout: Duration) -> String {
    let mut total_ms = 0.0;

    for i in 0..reps {
        eprint!("\r\t\tStarting input {input_file} ({}/{reps})", i + 1);
        stderr().flush().unwrap();
        let r = run_bin(bin, input_file, timeout);

        if r.is_err() {
            let e = r.unwrap_err();
            eprintln!("\n\t\t\t{e}");
            return e;
        } else {
            let stdout = r.unwrap();
            let ms = get_runtime_ms_from_stdout(&stdout);
            if ms.is_none() {
                eprintln!("\n\t\t\tCould not find time information in output.");
                return "NaN".to_string();
            }

            total_ms += ms.unwrap();
        }
    }
    eprint!("\n");

    return (total_ms / (reps as f32)).to_string();
}
