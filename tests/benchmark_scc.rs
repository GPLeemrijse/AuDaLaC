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
            "(P=1.3/N)",
            vec![
                ("tests/benchmarks/SCC/testcases/random_fb_1000_1289.init", vec!["-N".to_string(), "NodeSet=1000".to_string()], 1000, 1289, 1000),
                ("tests/benchmarks/SCC/testcases/random_fb_3162_4157.init", vec!["-N".to_string(), "NodeSet=3162".to_string()], 3162, 4157, 3162),
                ("tests/benchmarks/SCC/testcases/random_fb_10000_13021.init", vec!["-N".to_string(), "NodeSet=10000".to_string()], 10000, 13021, 10000),
                ("tests/benchmarks/SCC/testcases/random_fb_31623_40761.init", vec!["-N".to_string(), "NodeSet=31623".to_string()], 31623, 40761, 31623),
                ("tests/benchmarks/SCC/testcases/random_fb_100000_129651.init", vec!["-N".to_string(), "NodeSet=100000".to_string()], 100000, 129651, 100000),
                ("tests/benchmarks/SCC/testcases/random_fb_316228_411169.init", vec!["-N".to_string(), "NodeSet=316228".to_string()], 316228, 411169, 316228),
                ("tests/benchmarks/SCC/testcases/random_fb_1000000_1301143.init", vec!["-N".to_string(), "NodeSet=1000000".to_string()], 1000000, 1301143, 1000000),
                ("tests/benchmarks/SCC/testcases/random_fb_3162278_4110178.init", vec!["-N".to_string(), "NodeSet=3162278".to_string()], 3162278, 4110178, 3162278),
                ("tests/benchmarks/SCC/testcases/random_fb_10000000_12996160.init", vec!["-N".to_string(), "NodeSet=10000000".to_string()], 10000000, 12996160, 10000000),
            ],
        ),
    ]
}

fn scc_col_graphs() -> Vec<TestCase<'static>> {
    vec![
        (
            "(P=1.3/N)",
            vec![
                ("tests/benchmarks/SCC/testcases/random_col_1000_1289.init", Vec::new(), 1000, 1289, 0),
                ("tests/benchmarks/SCC/testcases/random_col_3162_4157.init", Vec::new(), 3162, 4157, 0),
                ("tests/benchmarks/SCC/testcases/random_col_10000_13021.init", Vec::new(), 10000, 13021, 0),
                ("tests/benchmarks/SCC/testcases/random_col_31623_40761.init", Vec::new(), 31623, 40761, 0),
                ("tests/benchmarks/SCC/testcases/random_col_100000_129651.init", Vec::new(), 100000, 129651, 0),
                ("tests/benchmarks/SCC/testcases/random_col_316228_411169.init", Vec::new(), 316228, 411169, 0),
                ("tests/benchmarks/SCC/testcases/random_col_1000000_1301143.init", Vec::new(), 1000000, 1301143, 0),
                ("tests/benchmarks/SCC/testcases/random_col_3162278_4110178.init", Vec::new(), 3162278, 4110178, 0),
                ("tests/benchmarks/SCC/testcases/random_col_10000000_12996160.init", Vec::new(), 10000000, 12996160, 0),
            ],
        ),
    ]
}



