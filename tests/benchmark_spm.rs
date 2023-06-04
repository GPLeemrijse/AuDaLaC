use std::io::Write;
use std::fs::File;
use crate::benchmarks::compile_config;
use crate::benchmarks::bench_testcases;
use crate::common::*;
use regex::Regex;


mod common;
mod benchmarks;

const REPS : usize = 10;

#[test]
fn test_benchmark_spm() {
	let vec_of_vec_of_configs = vec![
		memorder_impact_configs(),
		voting_impact_configs(),
		block_size_impact_configs(),
		weak_ro_impact_configs(),
		div_strat_impact_configs(),
	];

	let configs = Config::union(&vec_of_vec_of_configs);

	let mut vec_configs : Vec<Config> = Vec::new();
	for c in configs {
		vec_configs.push(c.clone());
	}

	benchmark_spm_set(&vec_configs, &invar_inev_eat_set(), "spm");
}

fn invar_inev_eat_set() -> Vec<(&'static str, Vec<&'static str>)> {
	vec![
		("invariantly_inevitably_eat", vec![
				"tests/benchmarks/SPM/testcases/dining/dining_2.invariantly_inevitably_eat.init",
				"tests/benchmarks/SPM/testcases/dining/dining_4.invariantly_inevitably_eat.init",
				"tests/benchmarks/SPM/testcases/dining/dining_6.invariantly_inevitably_eat.init",
				"tests/benchmarks/SPM/testcases/dining/dining_8.invariantly_inevitably_eat.init",
				"tests/benchmarks/SPM/testcases/dining/dining_10.invariantly_inevitably_eat.init",
			]
		),
	]
}

fn fname2problemsize(in_file : &str) -> String {
	Regex::new(r"^tests/benchmarks/SPM/testcases/dining/dining_([0-9]+)\.\w+\.init$")
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
		format!("{},algorithm,problem_type,problem_size,runtime\n", Config::HEADER)
			.as_bytes()
	).expect("Could not write header.");

	eprintln!("Benchmarking SPM ({set_name})...");

	for c in configs {
		let compile_err = compile_config(
			"SPM",
			"tests/benchmarks/SPM",
			None,
			c
		).err();
		
		if let Some(e) = compile_err {
			eprintln!("{}", e);
			continue;
		}
		let csv_prefix = format!("{},SPM", c.as_csv_row());
		bench_testcases(testcases, "tests/benchmarks/SPM/SPM.out", &csv_prefix, fname2problemsize, REPS, &mut result_file);
	}
}

