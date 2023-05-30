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
        .status()
        .map_or(false, |s| s.success())
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
    pub name: Option<&'a str>
}

impl<'a> Config<'_> {
    pub const HEADER : &str = "config,memorder,voting-strat,tpb,ipt";

    pub fn new(m : &'a str, v : &'a str, t : &'a str, i : &'a str, n : Option<&'a str>) -> Config<'a> {
        Config {
            memorder: m,
            voting: v,
            tpb: t,
            ipt: i,
            name: n
        }
    }

    pub fn to_args(&self) -> Vec<&str> {
        vec![
            "-m", self.memorder,
            "-v", self.voting,
            "-T", self.tpb,
            "-M", self.ipt
        ]
    }

    pub fn as_csv_row(&self) -> String {
        format!("{},{},{},{},{}",
            self,
            self.memorder,
            self.voting,
            self.tpb,
            self.ipt,
        )
    }

    pub fn cartesian(orders: &[&'static str], voting_strat: &[&'static str], tpb: &[&'static str], ipt: &[&'static str]) -> Vec<Config<'static>> {
        let mut result : Vec<Config> = Vec::new();

        for o in orders {
            for v in voting_strat {
                for t in tpb {
                    for i in ipt {
                        result.push(Config::new(o, v, t, i, None));
                    }
                }
            }
        }

        result
    }

    pub fn intersection(configs: &'a Vec<Vec<Config<'a>>>) -> HashSet<&'a Config<'a>>{
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