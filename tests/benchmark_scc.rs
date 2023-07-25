use crate::common::*;
mod common;

#[test]
fn test_benchmark_scc_fb() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<&Config> = Vec::new();
    for c in &configs {
        vec_configs.push(c);
    }
    
    let pgs = scc_fb_graphs();
    let tests : Vec<(&Config, &Vec<TestCase>)> = configs.iter()
                                                        .map(|c| 
                                                            (*c, &pgs)
                                                        )
                                                        .collect();

    benchmark(&tests, "SCC_FB", "tests/benchmarks/SCC");
}

#[test]
fn test_benchmark_scc_col() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<&Config> = Vec::new();
    for c in &configs {
        vec_configs.push(c);
    }
    
    let pgs = scc_col_graphs();
    let tests : Vec<(&Config, &Vec<TestCase>)> = configs.iter()
                                                        .map(|c| 
                                                            (*c, &pgs)
                                                        )
                                                        .collect();

    benchmark(&tests, "SCC_COL", "tests/benchmarks/SCC");
}

fn scc_fb_graphs() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random Graph (p=1.3/n)",
            vec![
                ("tests/benchmarks/SCC/testcases/random_scc_1000_1301.init", vec!["-N".to_string(), "NodeSet=1000".to_string()], 1000 + 1301),
                ("tests/benchmarks/SCC/testcases/random_scc_3162_4073.init", vec!["-N".to_string(), "NodeSet=3162".to_string()], 3162 + 4073),
                ("tests/benchmarks/SCC/testcases/random_scc_10000_13124.init", vec!["-N".to_string(), "NodeSet=10000".to_string()], 10000 + 13124),
                ("tests/benchmarks/SCC/testcases/random_scc_31623_40798.init", vec!["-N".to_string(), "NodeSet=31623".to_string()], 31623 + 40798),
                ("tests/benchmarks/SCC/testcases/random_scc_100000_129454.init", vec!["-N".to_string(), "NodeSet=100000".to_string()], 100000 + 129454),
                ("tests/benchmarks/SCC/testcases/random_scc_316228_410483.init", vec!["-N".to_string(), "NodeSet=316228".to_string()], 316228 + 410483),
                ("tests/benchmarks/SCC/testcases/random_scc_1000000_1299394.init", vec!["-N".to_string(), "NodeSet=1000000".to_string()], 1000000 + 1299394),
                ("tests/benchmarks/SCC/testcases/random_scc_3162278_4108430.init", vec!["-N".to_string(), "NodeSet=3162278".to_string()], 3162278 + 4108430),
                ("tests/benchmarks/SCC/testcases/random_scc_10000000_12994496.init", vec!["-N".to_string(), "NodeSet=10000000".to_string()], 10000000 + 12994496),
            ],
        ),
    ]
}

fn scc_col_graphs() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random Graph (p=1.3/n)",
            vec![
                ("tests/benchmarks/SCC/testcases/random_col_1000_1301.init", Vec::new(), 1000 + 1301),
                ("tests/benchmarks/SCC/testcases/random_col_3162_4073.init", Vec::new(), 3162 + 4073),
                ("tests/benchmarks/SCC/testcases/random_col_10000_13124.init", Vec::new(), 10000 + 13124),
                ("tests/benchmarks/SCC/testcases/random_col_31623_40798.init", Vec::new(), 31623 + 40798),
                ("tests/benchmarks/SCC/testcases/random_col_100000_129454.init", Vec::new(), 100000 + 129454),
                ("tests/benchmarks/SCC/testcases/random_col_316228_410483.init", Vec::new(), 316228 + 410483),
                ("tests/benchmarks/SCC/testcases/random_col_1000000_1299394.init", Vec::new(), 1000000 + 1299394),
                ("tests/benchmarks/SCC/testcases/random_col_3162278_4108430.init", Vec::new(), 3162278 + 4108430),
                ("tests/benchmarks/SCC/testcases/random_col_10000000_12994496.init", Vec::new(), 10000000 + 12994496),
            ],
        ),
    ]
}