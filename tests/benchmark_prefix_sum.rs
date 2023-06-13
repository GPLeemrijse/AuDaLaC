use std::time::Duration;
use crate::benchmarks::bench_testcases;
use crate::benchmarks::compile_config;
use crate::common::*;
use regex::Regex;
use std::fs::File;
use std::io::Write;

mod benchmarks;
mod common;

const REPS: usize = 10;

#[test]
fn test_benchmark_prefix_sum() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
        block_size_impact_configs(),
        weak_ro_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<Config> = Vec::new();
    for c in configs {
        vec_configs.push(c.clone());
    }

    benchmark_prefix_sum_set(&vec_configs, "prefix_sum");
}

fn random_prefix_sum_set() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "random_list",
            vec![
                "tests/benchmarks/prefix_sum/testcases/prefix_sum_10000.init",
                "tests/benchmarks/prefix_sum/testcases/prefix_sum_100000.init",
                "tests/benchmarks/prefix_sum/testcases/prefix_sum_1000000.init",
                "tests/benchmarks/prefix_sum/testcases/prefix_sum_10000000.init",
            ],
        ),
    ]
}


fn fname2nrof_elems(in_file: &str) -> String {
    Regex::new(r"^tests/benchmarks/prefix_sum/testcases/prefix_sum_([0-9]+)\.init$")
        .unwrap()
        .captures(in_file)
        .unwrap()
        .get(1)
        .unwrap()
        .as_str()
        .to_string()
}

fn benchmark_prefix_sum_set(configs: &Vec<Config>, set_name: &str) {
    if !is_benchmarking() {
        eprintln!("@@@@@ SKIPPING {set_name} @@@@@");
        return;
    }

    let mut result_file = File::create(format!("tests/benchmarks/prefix_sum/{set_name}_results.csv"))
        .expect("Could not create prefix_sum benchmark csv file.");
    result_file
        .write_all(
            format!(
                "{},algorithm,problem_type,problem_size,runtime\n",
                Config::HEADER
            )
            .as_bytes(),
        )
        .expect("Could not write header.");

    eprintln!("Benchmarking prefix_sum ({set_name})...");

    for (idx, c) in configs.iter().enumerate() {
        eprintln!("\tTesting prefix_sum config {}/{}: {c}", idx + 1, configs.len());
        let compile_err = compile_config(
            "prefix_sum",
            "tests/benchmarks/prefix_sum",
            Some("all"),
            c,
            Vec::new(),
        )
        .err();

        if let Some(e) = compile_err {
            eprintln!("{}", e);
            continue;
        }
        let csv_prefix = format!("{},prefix_sum", c.as_csv_row());
        bench_testcases(
            &random_prefix_sum_set(),
            "tests/benchmarks/prefix_sum/prefix_sum.out",
            &csv_prefix,
            fname2nrof_elems,
            REPS,
            &mut result_file,
            Duration::from_secs(10)
        );
    }
}
