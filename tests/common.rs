use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fmt::Display;
use std::process::Command;

pub fn is_benchmarking() -> bool {
    if let Ok(v) = env::var("BENCHMARK") {
        return v == "true";
    }
    return false;
}

pub fn is_nvcc_installed() -> bool {
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map_or(false, |s| s.status.success())
}

pub fn memorder_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed", "seqcons", "acqrel"];
    let voting_strat = ["naive-alternating"];
    let tpb = ["128", "512"];
    let w = ["0", "1"];

    Config::cartesian(&orders, &voting_strat, &tpb, &w)
}

pub fn voting_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive", "naive-alternating"];
    let tpb = ["128", "512"];
    let w = ["1"];

    Config::cartesian(&orders, &voting_strat, &tpb, &w)
}

pub fn weak_ro_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive-alternating"];
    let tpb = ["128", "512"];
    let w = ["0", "1"];

    Config::cartesian(&orders, &voting_strat, &tpb, &w)
}

pub fn block_size_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive-alternating"];
    let tpb = ["64", "128", "256", "512", "1024"];
    let w = ["1"];

    Config::cartesian(&orders, &voting_strat, &tpb, &w)
}

#[derive(Eq, PartialEq, Hash, Clone)]
pub struct Config<'a> {
    pub memorder: &'a str,
    pub voting: &'a str,
    pub tpb: &'a str,
    pub weak_non_racing: &'a str,
    pub name: Option<&'a str>,
}

impl<'a> Config<'_> {
    pub const HEADER: &str = "config,memorder,voting-strat,tpb,weak_non_racing";

    pub fn new(m: &'a str, v: &'a str, t: &'a str, w: &'a str, n: Option<&'a str>) -> Config<'a> {
        Config {
            memorder: m,
            voting: v,
            tpb: t,
            weak_non_racing: w,
            name: n,
        }
    }

    pub fn to_args(&self) -> Vec<String> {
        vec![
            "-m".to_string(),
            self.memorder.to_string(),
            "-v".to_string(),
            self.voting.to_string(),
            "-T".to_string(),
            self.tpb.to_string(),
            "-w".to_string(),
            self.weak_non_racing.to_string(),
        ]
    }

    pub fn as_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{}",
            self, self.memorder, self.voting, self.tpb, self.weak_non_racing,
        )
    }

    pub fn cartesian(
        orders: &[&'static str],
        voting_strat: &[&'static str],
        tpb: &[&'static str],
        weak_non_racing: &[&'static str],
    ) -> Vec<Config<'static>> {
        let mut result: Vec<Config> = Vec::new();

        for o in orders {
            for v in voting_strat {
                for t in tpb {
                    for w in weak_non_racing {
                        result.push(Config::new(o, v, t, w, None));
                    }
                }
            }
        }

        result
    }

    pub fn union(configs: &'a Vec<Vec<Config<'a>>>) -> HashSet<&'a Config<'a>> {
        let flat: Vec<&Config<'a>> = configs.iter().map(|v| v.iter()).flatten().collect();

        HashSet::from_iter(flat)
    }
}

impl Display for Config<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tuple = format!(
            "{}-{}-{}-{}",
            self.memorder, self.voting, self.tpb, self.weak_non_racing
        );
        write!(f, "{}", self.name.unwrap_or(&tuple))
    }
}
