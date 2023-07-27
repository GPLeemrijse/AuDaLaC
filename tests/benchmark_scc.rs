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
                ("tests/benchmarks/SCC/testcases/random_scc_1000_1309.init", vec!["-N".to_string(), "NodeSet=3000".to_string()], 1000, 1309, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_3162_4130.init", vec!["-N".to_string(), "NodeSet=9486".to_string()], 3162, 4130, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_10000_13061.init", vec!["-N".to_string(), "NodeSet=30000".to_string()], 10000, 13061, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_31623_41030.init", vec!["-N".to_string(), "NodeSet=94869".to_string()], 31623, 41030, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_100000_130295.init", vec!["-N".to_string(), "NodeSet=300000".to_string()], 100000, 130295, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_316228_411280.init", vec!["-N".to_string(), "NodeSet=948684".to_string()], 316228, 411280, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_1000000_1299887.init", vec!["-N".to_string(), "NodeSet=3000000".to_string()], 1000000, 1299887, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_3162278_4108274.init", vec!["-N".to_string(), "NodeSet=9486834".to_string()], 3162278, 4108274, 0),
                ("tests/benchmarks/SCC/testcases/random_scc_10000000_12995172.init", vec!["-N".to_string(), "NodeSet=30000000".to_string()], 10000000, 12995172, 0),
            ],
        ),
    ]
}

fn scc_col_graphs() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random Graph (p=1.3/n)",
            vec![
                ("tests/benchmarks/SCC/testcases/random_col_1000_1309.init", Vec::new(), 1000, 1309, 0),
                ("tests/benchmarks/SCC/testcases/random_col_3162_4130.init", Vec::new(), 3162, 4130, 0),
                ("tests/benchmarks/SCC/testcases/random_col_10000_13061.init", Vec::new(), 10000, 13061, 0),
                ("tests/benchmarks/SCC/testcases/random_col_31623_41030.init", Vec::new(), 31623, 41030, 0),
                ("tests/benchmarks/SCC/testcases/random_col_100000_130295.init", Vec::new(), 100000, 130295, 0),
                ("tests/benchmarks/SCC/testcases/random_col_316228_411280.init", Vec::new(), 316228, 411280, 0),
                ("tests/benchmarks/SCC/testcases/random_col_1000000_1299887.init", Vec::new(), 1000000, 1299887, 0),
                ("tests/benchmarks/SCC/testcases/random_col_3162278_4108274.init", Vec::new(), 3162278, 4108274, 0),
                ("tests/benchmarks/SCC/testcases/random_col_10000000_12995172.init", Vec::new(), 10000000, 12995172, 0),
            ],
        ),
    ]
}

