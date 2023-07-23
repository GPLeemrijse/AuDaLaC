use crate::common::*;
mod common;

#[test]
fn test_benchmark_spm() {
    let vec_of_vec_of_configs = vec![
        memorder_impact_configs(),
        voting_impact_configs(),
    ];

    let configs = Config::union(&vec_of_vec_of_configs);

    let mut vec_configs: Vec<&Config> = Vec::new();
    for c in &configs {
        vec_configs.push(c);
    }
    
    let pgs = parity_games();
    let tests : Vec<(&Config, &Vec<TestCase>)> = configs.iter()
                                                        .map(|c| 
                                                            (*c, &pgs)
                                                        )
                                                        .collect();

    benchmark(&tests, "SPM", "tests/benchmarks/SPM");
}

fn parity_games() -> Vec<TestCase<'static>> {
    vec![
        (
            "Invariantly Inevitably Eat",
            vec![
                // ("tests/benchmarks/SPM/testcases/dining/dining_4.invariantly_inevitably_eat.init", Vec::new(), 239 + 555),
                ("tests/benchmarks/SPM/testcases/dining/dining_5.invariantly_inevitably_eat.init", Vec::new(), 787 + 2177),
                // ("tests/benchmarks/SPM/testcases/dining/dining_6.invariantly_inevitably_eat.init", Vec::new(), 2597 + 8417),
                ("tests/benchmarks/SPM/testcases/dining/dining_7.invariantly_inevitably_eat.init", Vec::new(), 8575 + 32102),
                // ("tests/benchmarks/SPM/testcases/dining/dining_8.invariantly_inevitably_eat.init", Vec::new(), 28319 + 120943),
                ("tests/benchmarks/SPM/testcases/dining/dining_9.invariantly_inevitably_eat.init", Vec::new(), 93529 + 450679),
                // ("tests/benchmarks/SPM/testcases/dining/dining_10.invariantly_inevitably_eat.init", Vec::new(), 1663133 + 308903),
            ],
        ),
        (
            "Invariantly Plato Starves",
            vec![
                // ("tests/benchmarks/SPM/testcases/dining/dining_4.invariantly_plato_starves.init", Vec::new(), 504 + 174),
                ("tests/benchmarks/SPM/testcases/dining/dining_5.invariantly_plato_starves.init", Vec::new(), 570 + 2018),
                // ("tests/benchmarks/SPM/testcases/dining/dining_6.invariantly_plato_starves.init", Vec::new(), 1878 + 7854),
                ("tests/benchmarks/SPM/testcases/dining/dining_7.invariantly_plato_starves.init", Vec::new(), 6198 + 29888),
                // ("tests/benchmarks/SPM/testcases/dining/dining_8.invariantly_plato_starves.init", Vec::new(), 20466 + 111774),
                ("tests/benchmarks/SPM/testcases/dining/dining_9.invariantly_plato_starves.init", Vec::new(), 67590 + 412322),
                // ("tests/benchmarks/SPM/testcases/dining/dining_10.invariantly_plato_starves.init", Vec::new(), 1504368 + 223230),
            ],
        ),
        // (
        //     "Plato Infinitely Often Can Eat",
        //     vec![
        //         ("tests/benchmarks/SPM/testcases/dining/dining_4.plato_infinitely_often_can_eat.init", Vec::new()),
        //         ("tests/benchmarks/SPM/testcases/dining/dining_6.plato_infinitely_often_can_eat.init", Vec::new()),
        //         ("tests/benchmarks/SPM/testcases/dining/dining_8.plato_infinitely_often_can_eat.init", Vec::new()),
        //         ("tests/benchmarks/SPM/testcases/dining/dining_10.plato_infinitely_often_can_eat.init", Vec::new()),
        //     ],
        // ),
        // (
        //     "Invariantly Possibly Eat",
        //     vec![
        //         ("tests/benchmarks/SPM/testcases/dining/dining_4.invariantly_possibly_eat.init", Vec::new()),
        //         ("tests/benchmarks/SPM/testcases/dining/dining_6.invariantly_possibly_eat.init", Vec::new()),
        //         ("tests/benchmarks/SPM/testcases/dining/dining_8.invariantly_possibly_eat.init", Vec::new()),
        //         ("tests/benchmarks/SPM/testcases/dining/dining_10.invariantly_possibly_eat.init", Vec::new()),
        //     ],
        // ),
    ]
}