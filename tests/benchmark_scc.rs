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
                ("tests/benchmarks/SCC/testcases/random_scc_5000_6512.init", vec!["-N".to_string(), "NodeSet=5000".to_string()], 5000 + 6512),
                ("tests/benchmarks/SCC/testcases/random_scc_50000_64814.init", vec!["-N".to_string(), "NodeSet=50000".to_string()], 50000 + 64814),
                ("tests/benchmarks/SCC/testcases/random_scc_500000_649534.init", vec!["-N".to_string(), "NodeSet=500000".to_string()], 500000 + 649534),
                ("tests/benchmarks/SCC/testcases/random_scc_5000000_6502185.init", vec!["-N".to_string(), "NodeSet=5000000".to_string()], 5000000 + 6502185),
            ],
        ),
    ]
}

fn scc_col_graphs() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random Graph (p=1.3/n)",
            vec![
                ("tests/benchmarks/SCC/testcases/random_col_5000_6512.init", Vec::new(), 5000 + 6512),
                ("tests/benchmarks/SCC/testcases/random_col_50000_64814.init", Vec::new(), 50000 + 64814),
                ("tests/benchmarks/SCC/testcases/random_col_500000_649534.init", Vec::new(), 500000 + 649534),
                ("tests/benchmarks/SCC/testcases/random_col_5000000_6502185.init", Vec::new(), 5000000 + 6502185),
            ],
        ),
    ]
}