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
                ("tests/benchmarks/synthesis/testcases/synthesis_10000_12923.init", Vec::new(), 10000 + 12923),
                ("tests/benchmarks/synthesis/testcases/synthesis_31623_41401.init", Vec::new(), 31623 + 41401),
                ("tests/benchmarks/synthesis/testcases/synthesis_100000_129684.init", Vec::new(), 100000 + 129684),
                ("tests/benchmarks/synthesis/testcases/synthesis_316228_410964.init", Vec::new(), 316228 + 410964),
                ("tests/benchmarks/synthesis/testcases/synthesis_1000000_1300269.init", Vec::new(), 1000000 + 1300269),
                ("tests/benchmarks/synthesis/testcases/synthesis_3162278_4112232.init", Vec::new(), 3162278 + 4112232),
                ("tests/benchmarks/synthesis/testcases/synthesis_10000000_13006610.init", Vec::new(), 10000000 + 13006610),
                //("tests/benchmarks/synthesis/testcases/synthesis_31622777_41100055.init", Vec::new(), 31622777 + 41100055)
            ],
        )
    ]
}