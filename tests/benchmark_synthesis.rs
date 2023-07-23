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
                // ("tests/benchmarks/synthesis/testcases/synthesis_10000_13030.init", Vec::new(),   10000 + 13030),
                ("tests/benchmarks/synthesis/testcases/synthesis_55000_71400.init", Vec::new(),   55000 + 71400),
                // ("tests/benchmarks/synthesis/testcases/synthesis_100000_129856.init", Vec::new(),  100000 + 129856),
                ("tests/benchmarks/synthesis/testcases/synthesis_550000_713755.init", Vec::new(),   550000 + 713755),
                // ("tests/benchmarks/synthesis/testcases/synthesis_1000000_1298699.init", Vec::new(), 1000000 + 1298699),
                ("tests/benchmarks/synthesis/testcases/synthesis_5500000_7153328.init", Vec::new(),   5500000 + 7153328),
                // ("tests/benchmarks/synthesis/testcases/synthesis_10000000_13001871.init", Vec::new(),10000000 + 13001871),
            ],
        )
    ]
}