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
            "random_0.005",
            vec![
                "tests/benchmarks/SCC/testcases/random_scc_1000_5076_0.005.init",
                "tests/benchmarks/SCC/testcases/random_scc_2500_30877_0.005.init",
                "tests/benchmarks/SCC/testcases/random_scc_5000_124845_0.005.init",
                "tests/benchmarks/SCC/testcases/random_scc_10000_500574_0.005.init",
            ],
        ),
        (
            "random_0.01",
            vec![
                "tests/benchmarks/SCC/testcases/random_scc_1000_9813_0.01.init",
                "tests/benchmarks/SCC/testcases/random_scc_2500_62582_0.01.init",
                "tests/benchmarks/SCC/testcases/random_scc_5000_250467_0.01.init",
                "tests/benchmarks/SCC/testcases/random_scc_10000_999898_0.01.init",
            ],
        ),
        (
            "random_0.1",
            vec![
                "tests/benchmarks/SCC/testcases/random_scc_1000_100095_0.1.init",
                "tests/benchmarks/SCC/testcases/random_scc_2500_624033_0.1.init",
                "tests/benchmarks/SCC/testcases/random_scc_5000_2500146_0.1.init",
                "tests/benchmarks/SCC/testcases/random_scc_10000_10001053_0.1.init",
            ],
        ),
    ]
}

fn random_mp_set() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "random_0.005",
            vec![
                "tests/benchmarks/SCC/testcases/random_mp_1000_5076_0.005.init",
                "tests/benchmarks/SCC/testcases/random_mp_2500_30877_0.005.init",
                "tests/benchmarks/SCC/testcases/random_mp_5000_124845_0.005.init",
                "tests/benchmarks/SCC/testcases/random_mp_10000_500574_0.005.init",
            ],
        ),
        (
            "random_0.01",
            vec![
                "tests/benchmarks/SCC/testcases/random_mp_1000_9813_0.01.init",
                "tests/benchmarks/SCC/testcases/random_mp_2500_62582_0.01.init",
                "tests/benchmarks/SCC/testcases/random_mp_5000_250467_0.01.init",
                "tests/benchmarks/SCC/testcases/random_mp_10000_999898_0.01.init",
            ],
        ),
        (
            "random_0.1",
            vec![
                "tests/benchmarks/SCC/testcases/random_mp_1000_100095_0.1.init",
                "tests/benchmarks/SCC/testcases/random_mp_2500_624033_0.1.init",
                "tests/benchmarks/SCC/testcases/random_mp_5000_2500146_0.1.init",
                "tests/benchmarks/SCC/testcases/random_mp_10000_10001053_0.1.init",
            ],
        ),
    ]
}

fn fname2nrof_edges(in_file: &str) -> String {
    Regex::new(r"^tests/benchmarks/SCC/testcases/random_\w+_[0-9]+_([0-9]+)_.+\.init$")
        .unwrap()
        .captures(in_file)
        .unwrap()
        .get(1)
        .unwrap()
        .as_str()
        .to_string()
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
            vec!["-N".to_string(), "NodeSet=10001".to_string()],
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
        );
    }
}
