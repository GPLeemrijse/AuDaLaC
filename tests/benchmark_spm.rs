use std::io::Write;
use std::fs::File;
use crate::benchmarks::compile_config;
use crate::benchmarks::bench_testcases;
use crate::common::*;
use std::path::Path;
use regex::Regex;
use std::process::Command;

mod common;
mod benchmarks;

const REPS : usize = 10;

#[test]
fn test_impact_of_memorder() {
	benchmark_spm_set(&memorder_impact_configs(), &invar_inev_eat_set(), "memorder_impact");
}

#[test]
fn test_impact_of_voting() {
	benchmark_spm_set(&voting_impact_configs(), &invar_inev_eat_set(), "voting_impact");
}

#[test]
fn test_impact_of_block_size() {
	benchmark_spm_set(&block_size_impact_configs(), &invar_inev_eat_set(), "blocksize_impact");
}


fn memorder_impact_configs() -> Vec<Config<'static>> {
	let orders = ["relaxed", "seqcons", "acqrel"];
	let voting_strat = ["naive-alternating"];
	let tpb = ["128", "512"];
	let ipt = ["4", "32"];

	Config::cartesian(&orders, &voting_strat, &tpb, &ipt)
}

fn voting_impact_configs() -> Vec<Config<'static>> {
	let orders = ["relaxed"];
	let voting_strat = ["naive", "naive-alternating"];
	let tpb = ["128", "512"];
	let ipt = ["4", "32"];

	Config::cartesian(&orders, &voting_strat, &tpb, &ipt)
}

fn block_size_impact_configs() -> Vec<Config<'static>> {
	let orders = ["relaxed"];
	let voting_strat = ["naive-alternating"];
	let tpb = ["64", "128", "256", "512", "1024"];
	let ipt = ["1", "4", "16", "32"];

	Config::cartesian(&orders, &voting_strat, &tpb, &ipt)
}

fn invar_inev_eat_set() -> Vec<(&'static str, Vec<&'static str>)> {
	vec![
		("invariantly_inevitably_eat", vec![
				"dining/dining_2.invariantly_inevitably_eat.init",
				"dining/dining_4.invariantly_inevitably_eat.init",
				"dining/dining_6.invariantly_inevitably_eat.init",
				"dining/dining_8.invariantly_inevitably_eat.init",
				"dining/dining_10.invariantly_inevitably_eat.init",
			]
		),
	]
}


fn run_spm(file : &str) -> Result<String, String> {
	if is_nvcc_installed() {
		let r = Command::new("tests/benchmarks/SPM/SPM.out")
	        .arg(Path::new("tests/benchmarks/SPM/testcases/").join(file))
	        .output();
	    if r.is_err() {
			return Err("Binary would not run.".to_string());
		}

		let out = r.unwrap();
		if !out.status.success() {
			return Err("non-zero exitcode.".to_string());
		}
		Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        Err("skipped".to_string())
    }
}

fn fname2problemsize(in_file : &str) -> String {
	Regex::new(r"^dining/dining_([0-9]+)\.\w+\.init$")
		.unwrap()
		.captures(in_file)
		.unwrap()
		.get(1)
		.unwrap()
		.as_str()
		.to_string()
}



fn benchmark_spm_set(configs: &Vec<Config>, testcases: &Vec<(&str, Vec<&str>)>, set_name: &str) {
	if !is_benchmarking() {
		eprintln!("@@@@@ SKIPPING {set_name} @@@@@");
		return;
	}

	let mut result_file = File::create(format!("tests/benchmarks/SPM/{set_name}_results.csv")).expect("Could not create SPM benchmark csv file.");
	result_file.write_all(
		format!("{},formula,problemsize,runtime\n", Config::HEADER)
			.as_bytes()
	).expect("Could not write header.");

	eprintln!("Benchmarking SPM ({set_name})...");

	for c in configs {
		let compile_err = compile_config(
			"SPM",
			"tests/benchmarks/SPM",
			c
		).err();
		
		if let Some(e) = compile_err {
			eprintln!("{}", e);
			continue;
		}
		let csv_prefix = c.as_csv_row();
		bench_testcases(testcases, &csv_prefix, run_spm, fname2problemsize, REPS, &mut result_file);
	}
}

