use std::collections::HashSet;
use std::fmt::Display;
use std::fmt;
use std::process::Command;
use std::env;

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
    let ipt = ["4", "32"];
    let w = ["0"];
    let d = ["gridsize"];

    Config::cartesian(&orders, &voting_strat, &tpb, &ipt, &w, &d)
}

pub fn voting_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive", "naive-alternating"];
    let tpb = ["128", "512"];
    let ipt = ["4", "32"];
    let w = ["0"];
    let d = ["gridsize"];

    Config::cartesian(&orders, &voting_strat, &tpb, &ipt, &w, &d)
}

pub fn weak_ro_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive-alternating"];
    let tpb = ["128", "512"];
    let ipt = ["4", "32"];
    let w = ["0", "1"];
    let d = ["gridsize"];

    Config::cartesian(&orders, &voting_strat, &tpb, &ipt, &w, &d)
}

pub fn div_strat_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive-alternating"];
    let tpb = ["128", "512"];
    let ipt = ["4", "32"];
    let w = ["1"];
    let d = ["blocksize", "gridsize"];

    Config::cartesian(&orders, &voting_strat, &tpb, &ipt, &w, &d)
}

pub fn block_size_impact_configs() -> Vec<Config<'static>> {
    let orders = ["relaxed"];
    let voting_strat = ["naive-alternating"];
    let tpb = ["64", "128", "256", "512", "1024"];
    let ipt = ["1", "4", "16", "32"];
    let w = ["1"];
    let d = ["gridsize"];

    Config::cartesian(&orders, &voting_strat, &tpb, &ipt, &w, &d)
}


#[derive(Eq)]
#[derive(PartialEq)]
#[derive(Hash)]
#[derive(Clone)]
pub struct Config<'a> {
    pub memorder: &'a str,
    pub voting: &'a str,
    pub tpb: &'a str,
    pub ipt: &'a str,
    pub weak_ro: &'a str,
    pub d_strat: &'a str,
    pub name: Option<&'a str>
}

impl<'a> Config<'_> {
    pub const HEADER : &str = "config,memorder,voting-strat,tpb,ipt,weak_read_only";

    pub fn new(
        m : &'a str,
        v : &'a str,
        t : &'a str,
        i : &'a str,
        w: &'a str,
        d: &'a str,
        n : Option<&'a str>
    ) -> Config<'a> {
        Config {
            memorder: m,
            voting: v,
            tpb: t,
            ipt: i,
            weak_ro: w,
            d_strat: d,
            name: n
        }
    }

    pub fn to_args(&self) -> Vec<String> {
        vec![
            "-m".to_string(), self.memorder.to_string(),
            "-v".to_string(), self.voting.to_string(),
            "-T".to_string(), self.tpb.to_string(),
            "-M".to_string(), self.ipt.to_string(),
            "-w".to_string(), self.weak_ro.to_string(),
            "-d".to_string(), self.d_strat.to_string(),
        ]
    }

    pub fn as_csv_row(&self) -> String {
        format!("{},{},{},{},{},{},{}",
            self,
            self.memorder,
            self.voting,
            self.tpb,
            self.ipt,
            self.weak_ro,
            self.d_strat
        )
    }

    pub fn cartesian(
        orders: &[&'static str],
        voting_strat: &[&'static str],
        tpb: &[&'static str],
        ipt: &[&'static str],
        weak_ro: &[&'static str],
        d_strat: &[&'static str],
    ) -> Vec<Config<'static>> {
        let mut result : Vec<Config> = Vec::new();

        for o in orders {
        for v in voting_strat {
        for t in tpb {
        for i in ipt {
        for w in weak_ro {
        for d in d_strat {
            result.push(Config::new(o, v, t, i, w, d, None));
        }}}}}}

        result
    }

    pub fn union(configs: &'a Vec<Vec<Config<'a>>>) -> HashSet<&'a Config<'a>>{
        let flat : Vec<&Config<'a>> = configs.iter()
                                             .map(|v|
                                                v.iter()
                                             )
                                             .flatten()
                                             .collect();
        
        HashSet::from_iter(
            flat
        )
    }
}

impl Display for Config<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let tuple = format!("{}-{}-{}-{}",self.memorder, self.voting, self.tpb, self.ipt);
        write!(f, "{}", self.name.unwrap_or(&tuple))
    }
}