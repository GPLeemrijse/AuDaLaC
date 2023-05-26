use std::io::stdout;
use std::io::Write;
use std::fs::File;
use regex::Regex;
use std::process::Output;
use std::path::Path;
use crate::common::run_benchmarks;
use crate::common::run_make;
use crate::common::run_compiler;
use std::process::Command;

mod common;

const REPS : usize = 10;

const CONFIGS: [(&str, &str, &str, &str); 19] = [
	("memorder", "voting-strat", "tpb", "ipt"),
	("relaxed", "naive-alternating", "128", "1"),
	("relaxed", "naive-alternating", "128", "16"),
	("relaxed", "naive-alternating", "128", "32"),

	("relaxed", "naive-alternating", "256", "1"),
	("relaxed", "naive-alternating", "256", "16"),
	("relaxed", "naive-alternating", "256", "32"),

	("relaxed", "naive-alternating", "512", "1"),
	("relaxed", "naive-alternating", "512", "16"),
	("relaxed", "naive-alternating", "512", "32"),

	("seqcons", "naive-alternating", "128", "2"),
	("seqcons", "naive-alternating", "128", "4"),
	("seqcons", "naive-alternating", "128", "8"),

	("seqcons", "naive-alternating", "256", "2"),
	("seqcons", "naive-alternating", "256", "4"),
	("seqcons", "naive-alternating", "256", "8"),

	("seqcons", "naive-alternating", "512", "2"),
	("seqcons", "naive-alternating", "512", "4"),
	("seqcons", "naive-alternating", "512", "8"),
];

const TESTCASES: [(&str, [&str; 3]); 4] = [
	("invariantly_inevitably_eat", [
			"dining/dining_2.invariantly_inevitably_eat.init",
			"dining/dining_6.invariantly_inevitably_eat.init",
			"dining/dining_8.invariantly_inevitably_eat.init",
		]
	),

	("invariantly_plato_starves", [
			"dining/dining_2.invariantly_plato_starves.init",
			"dining/dining_6.invariantly_plato_starves.init",
			"dining/dining_8.invariantly_plato_starves.init",
		]
	),

	("invariantly_possibly_eat", [
			"dining/dining_2.invariantly_possibly_eat.init",
			"dining/dining_6.invariantly_possibly_eat.init",
			"dining/dining_8.invariantly_possibly_eat.init",
		]
	),
	
	("plato_infinitely_often_can_eat", [
			"dining/dining_2.plato_infinitely_often_can_eat.init",
			"dining/dining_6.plato_infinitely_often_can_eat.init",
			"dining/dining_8.plato_infinitely_often_can_eat.init",
		]
	)
];


fn run_spm(file : &str) -> Result<Output, std::io::Error> {
	Command::new("tests/benchmarks/SPM/SPM.out")
        .arg(Path::new("tests/benchmarks/SPM/testcases/").join(file))
        .output()
}

fn get_runtime_ms_from_stdout(output : &str) -> Option<f32> {
	let re = Regex::new(r"Total walltime GPU: ([0-9]+\.[0-9]+) ms").unwrap();
	for l in output.lines() {
		let ms = re.captures(l)
				   .map(|c| c.get(1)
				   			 .unwrap()
				   			 .as_str()
				   			 .parse::<f32>()
				   			 .unwrap());
		if ms.is_some() {
			return ms;
		}
	}
	return None;
}

fn adl_compile(config : &(&str, &str, &str, &str)) -> Result<(), String> {
	eprintln!("\t\tCompiling ADL code...");
	let r = run_compiler(vec![
		"tests/benchmarks/SPM/SPM.adl",
		"-o", "tests/benchmarks/SPM/SPM.cu",
		"-t",
		"-m", config.0,
		"-v", config.1,
		"-T", config.2,
		"-M", config.3
	]);

	if r.is_err() {
		Err("\t\t\tFailed to start ADL compiler.".to_string())
	} else if !r.unwrap().success() {
		Err("\t\t\tFailed to compile ADL code.".to_string())
	} else {
		Ok(())
	}
}

fn cuda_compile() -> Result<(), String> {
	eprintln!("\t\tCompiling CUDA code...");
	let r = run_make("tests/benchmarks/SPM");

	if r.is_err() {
		Err("\t\t\tFailed to start Make.".to_string())
	} else if !r.unwrap().success() {
		Err("\t\t\tFailed to compile CUDA code.".to_string())
	} else {
		Ok(())
	}
}

fn bench_config(config : &(&str, &str, &str, &str), result_file : &mut dyn Write) -> Result<(), String> {
	eprintln!("\tTesting config {:?}", config);

	adl_compile(config)?;
	cuda_compile()?;
	

	for (f_name, files) in TESTCASES {
		println!("\t\t\tTesting {f_name}...");
		for f in files {
			bench_file(config, f_name, f, result_file);
		}
	}
	Ok(())
}

fn bench_file(c : &(&str, &str, &str, &str), f_name : &str, f : &str, result_file : &mut dyn Write) {
	let ps = Regex::new(r"^dining/dining_([0-9]+)\.\w+\.init$")
				.unwrap()
				.captures(f)
				.unwrap()
				.get(1)
				.unwrap()
				.as_str();
	let mut result_str = None;
	let mut total_ms = 0.0;

	for i in 0..REPS {
		print!("\r\t\t\t\tStarting problem size {ps} ({}/{REPS})", i + 1);
		stdout().flush().unwrap();
		let r = run_spm(f);
		if r.is_err() {
			eprintln!("\n\t\t\t\t\tSPM.out would not run.");
			result_str = Some("SPM.out would not run.".to_string());
			break;
		}

		let out = r.unwrap();
		if !out.status.success() {
			eprintln!("\n\t\t\t\t\tSPM.out returned non-zero exit code.");
			result_str = Some("non-zero exitcode".to_string());
			break;
		}

		let ms = get_runtime_ms_from_stdout(&String::from_utf8_lossy(&out.stdout));
		if ms.is_none() {
			eprintln!("\n\t\t\t\t\tCould not find time information in output.");
			result_str = Some("NaN".to_string());
			break;
		}

		total_ms += ms.unwrap();
	}
	print!("\n");
	
	if result_str.is_none() {
		result_str = Some((total_ms / (REPS as f32)).to_string());
	}

	result_file.write_all(format!("{}, {}, {}, {}, {}, {}, {}\n",
		c.0,
		c.1,
		c.2,
		c.3,
		f_name,
		ps,
		result_str.unwrap()).as_bytes()
	).expect("Could not write result.");
}


#[test]
fn test_benchmark_spm() {
	if !run_benchmarks() {
		eprintln!("@@@@@ SKIPPING test_benchmark_spm @@@@@");
		return;
	}
	let mut result_file = File::create("tests/benchmarks/SPM/results.csv").expect("Could not create SPM benchmark csv file.");
	let header = &CONFIGS[0];

	result_file.write_all(format!("{}, {}, {}, {}, {}, {}, {}\n",
		header.0,
		header.1,
		header.2,
		header.3,
		"formula",
		"problem size",
		"runtime (ms)").as_bytes()
	).expect("Could not write header.");

	eprintln!("Benchmarking SPM...");
	for c in &CONFIGS[1..] {
		let bench_err = bench_config(c, &mut result_file).err();
		if let Some(e) = bench_err {
			eprintln!("{}", e);
		}
	}
}