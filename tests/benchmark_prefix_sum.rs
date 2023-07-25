use crate::common::*;
mod common;

#[test]
fn test_benchmark_prefix_sum() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<&Config> = Vec::new();
    for c in &configs {
        vec_configs.push(c);
    }
    
    let pgs = prefix_sum_files();
    let tests : Vec<(&Config, &Vec<TestCase>)> = configs.iter()
                                                        .map(|c| 
                                                            (*c, &pgs)
                                                        )
                                                        .collect();

    benchmark(&tests, "prefix_sum", "tests/benchmarks/prefix_sum");
}

fn prefix_sum_files() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random",
            vec![
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_1000.init", Vec::new(), 1000),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_3162.init", Vec::new(), 3162),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_10000.init", Vec::new(), 10000),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_31623.init", Vec::new(), 31623),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_100000.init", Vec::new(), 100000),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_316228.init", Vec::new(), 316228),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_1000000.init", Vec::new(), 1000000),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_3162278.init", Vec::new(), 3162278),
                ("tests/benchmarks/prefix_sum/testcases/prefix_sum_10000000.init", Vec::new(), 10000000),
            ],
        )
    ]
}
