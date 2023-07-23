use crate::common::*;
mod common;

#[test]
fn test_benchmark_sorting() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<&Config> = Vec::new();
    for c in &configs {
        vec_configs.push(c);
    }
    
    let pgs = sorting_files();
    let tests : Vec<(&Config, &Vec<TestCase>)> = configs.iter()
                                                        .map(|c| 
                                                            (*c, &pgs)
                                                        )
                                                        .collect();

    benchmark(&tests, "sorting", "tests/benchmarks/sorting");
}

fn sorting_files() -> Vec<TestCase<'static>> {
    vec![
        (
            "Random",
            vec![
                ("tests/benchmarks/sorting/testcases/sorting_10000.init", Vec::new(),   10000),
                ("tests/benchmarks/sorting/testcases/sorting_55000.init", Vec::new(),   55000),
                ("tests/benchmarks/sorting/testcases/sorting_100000.init", Vec::new(),  100000),
                ("tests/benchmarks/sorting/testcases/sorting_550000.init", Vec::new(),   550000),
                ("tests/benchmarks/sorting/testcases/sorting_1000000.init", Vec::new(), 1000000),
                ("tests/benchmarks/sorting/testcases/sorting_5500000.init", Vec::new(),   5500000),
                ("tests/benchmarks/sorting/testcases/sorting_10000000.init", Vec::new(),10000000),
            ],
        )
    ]
}