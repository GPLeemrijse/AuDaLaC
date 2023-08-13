use crate::common::*;
mod common;

#[test]
fn test_benchmark_synthesis() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<&Config> = Vec::new();
    for c in &configs {
        vec_configs.push(c);
    }
    
    let pgs = synthesis_files();
    let tests : Vec<(&Config, &Vec<TestCase>)> = configs.iter()
                                                        .map(|c| 
                                                            (*c, &pgs)
                                                        )
                                                        .collect();

    benchmark(&tests, "synthesis", "tests/benchmarks/synthesis");
}

fn synthesis_files() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random",
            vec![
                ("tests/benchmarks/synthesis/testcases/synthesis_1000_1280.init", Vec::new(), 1000, 768, 512),
                ("tests/benchmarks/synthesis/testcases/synthesis_3162_4098.init", Vec::new(), 3162, 2458, 1640),
                ("tests/benchmarks/synthesis/testcases/synthesis_10000_13116.init", Vec::new(), 10000, 7869, 5247),
                ("tests/benchmarks/synthesis/testcases/synthesis_31623_41591.init", Vec::new(), 31623, 24954, 16637),
                ("tests/benchmarks/synthesis/testcases/synthesis_100000_130338.init", Vec::new(), 100000, 78202, 52136),
                ("tests/benchmarks/synthesis/testcases/synthesis_316228_411059.init", Vec::new(), 316228, 246635, 164424),
                ("tests/benchmarks/synthesis/testcases/synthesis_1000000_1298826.init", Vec::new(), 1000000, 779295, 519531),
                ("tests/benchmarks/synthesis/testcases/synthesis_3162278_4114117.init", Vec::new(), 3162278, 2468470, 1645647),
                ("tests/benchmarks/synthesis/testcases/synthesis_10000000_12997394.init", Vec::new(), 10000000, 7798436, 5198958),
            ],
        )
    ]
}
