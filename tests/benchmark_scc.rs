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
fn test_benchmark_scc() {
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

    benchmark_scc_set(&vec_configs, "scc");
}

fn random_scc_set() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "random n*p=1.3",
            vec![
                "tests/benchmarks/SCC/testcases/random_scc_50000_65319.init",
                "tests/benchmarks/SCC/testcases/random_scc_500000_649925.init",
                "tests/benchmarks/SCC/testcases/random_scc_5000000_6499862.init",
            ],
        ),
    ]
}

fn random_mp_set() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "random n*p=1.3",
            vec![
                "tests/benchmarks/SCC/testcases/random_mp_50000_65319.init",
                "tests/benchmarks/SCC/testcases/random_mp_500000_649925.init",
                "tests/benchmarks/SCC/testcases/random_mp_5000000_6499862.init",
            ],
        ),
    ]
}

fn fname2nrof_edges(in_file: &str) -> String {
    let capts = Regex::new(r"^tests/benchmarks/SCC/testcases/random_\w+_([0-9]+)_([0-9]+)\.init$")
        .unwrap()
        .captures(in_file)
        .unwrap();

    let nodes = capts.get(1)
         .unwrap()
         .as_str().parse::<i32>().unwrap();

    let edges = capts.get(2)
         .unwrap()
         .as_str().parse::<i32>().unwrap();

    (nodes + edges).to_string()
}

fn benchmark_scc_set(configs: &Vec<Config>, set_name: &str) {
    if !is_benchmarking() {
        eprintln!("@@@@@ SKIPPING {set_name} @@@@@");
        return;
    }

    let mut result_file = File::create(format!("tests/benchmarks/SCC/{set_name}_results.csv"))
        .expect("Could not create SCC benchmark csv file.");
    result_file
        .write_all(
            format!(
                "{},algorithm,problem_type,problem_size,runtime\n",
                Config::HEADER
            )
            .as_bytes(),
        )
        .expect("Could not write header.");

    eprintln!("Benchmarking SCC ({set_name})...");

    for (idx, c) in configs.iter().enumerate() {
        eprintln!("\tTesting SCC config {}/{}: {c}", idx + 1, configs.len());
        let compile_err = compile_config(
            "SCC",
            "tests/benchmarks/SCC",
            Some("SCC"),
            c,
            vec!["-N".to_string(), "NodeSet=5000000".to_string()],
        )
        .err();

        if let Some(e) = compile_err {
            eprintln!("{}", e);
            continue;
        }
        let csv_prefix = format!("{},SCC", c.as_csv_row());
        bench_testcases(
            &random_scc_set(),
            "tests/benchmarks/SCC/SCC.out",
            &csv_prefix,
            fname2nrof_edges,
            REPS,
            &mut result_file,
            Duration::from_secs(60)
        );
    }

    for (idx, c) in configs.iter().enumerate() {
        eprintln!("\tTesting SCC_MP config {}/{}: {c}", idx + 1, configs.len());
        let compile_err =
            compile_config("SCC_MP", "tests/benchmarks/SCC", Some("MP"), c, Vec::new()).err();

        if let Some(e) = compile_err {
            eprintln!("{}", e);
            continue;
        }
        let csv_prefix = format!("{},SCC_MP", c.as_csv_row());
        bench_testcases(
            &random_mp_set(),
            "tests/benchmarks/SCC/SCC_MP.out",
            &csv_prefix,
            fname2nrof_edges,
            REPS,
            &mut result_file,
            Duration::from_secs(60)
        );
    }
}
