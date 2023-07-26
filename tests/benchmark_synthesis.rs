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
                ("tests/benchmarks/synthesis/testcases/synthesis_1000_1240.init", Vec::new(), 1000 + 1240),
                ("tests/benchmarks/synthesis/testcases/synthesis_3162_4027.init", Vec::new(), 3162 + 4027),
                ("tests/benchmarks/synthesis/testcases/synthesis_10000_12933.init", Vec::new(), 10000 + 12933),
                ("tests/benchmarks/synthesis/testcases/synthesis_31623_41167.init", Vec::new(), 31623 + 41167),
                ("tests/benchmarks/synthesis/testcases/synthesis_100000_130524.init", Vec::new(), 100000 + 130524),
                ("tests/benchmarks/synthesis/testcases/synthesis_316228_410968.init", Vec::new(), 316228 + 410968),
                ("tests/benchmarks/synthesis/testcases/synthesis_1000000_1299707.init", Vec::new(), 1000000 + 1299707),
                ("tests/benchmarks/synthesis/testcases/synthesis_3162278_4113503.init", Vec::new(), 3162278 + 4113503),
                ("tests/benchmarks/synthesis/testcases/synthesis_10000000_13005088.init", Vec::new(), 10000000 + 13005088),
            ],
        )
    ]
}
