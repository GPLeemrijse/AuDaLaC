use std::time::Instant;
use std::fs;
use std::str;
use std::env;

fn run_benchmarks() -> bool {
    if let Ok(v) = env::var("CI") {
        return v != "true";
    }
    return true;
}

#[test]
fn test_reachability_benchmarks() {
    if !run_benchmarks() {
        println!("Skipping benchmarks...");
        return;
    }

    let mut paths : Vec<String> = fs::read_dir("tests/benchmarks/reachability").unwrap().map(|p| p.as_ref().unwrap().path().display().to_string()).collect();
    let mut measurements : Vec<(u32, u32, f64, u32, u64, u64)> = Vec::new();
    let nrof_files = paths.len();
    paths.sort();

    for (idx, path) in paths.iter().enumerate() {
        if path.ends_with(".adl") {
            let last_slash = path.rfind('/').unwrap();
            let file_name = &path[last_slash+1..path.len()-4];
            let cu_file = format!("/tmp/{file_name}.cu");
            let bin_file = format!("/tmp/{file_name}.out");
            let parts: Vec<&str> = file_name.split("_").collect();
            let ps = parts[parts.len()-3].parse::<u32>().unwrap();
            let n = parts[parts.len()-2].parse::<u32>().unwrap();
            let m = parts[parts.len()-1].parse::<u32>().unwrap();

            println!("{}/{}: Compiling {} to CUDA...", idx + 1, nrof_files, file_name);
            let adl_comp_time = compile_adl_to_cuda(&path, &cu_file, n*m + 10);
            println!("Completed in {} seconds.", adl_comp_time);
            println!("Compiling {} to binary...", file_name);
            let bin_comp_time = compile_cuda_to_bin(&cu_file, &bin_file);
            println!("Completed in {} seconds.", bin_comp_time);
            println!("Running {} on GPU.", file_name);
            let (time, _, _) = time_binary(&bin_file, 100);
            println!("Completed in {} ms.\n", time);
            
            measurements.push((n as u32, m as u32, time, ps as u32, adl_comp_time, bin_comp_time));
        }
    }

    let mut csv = String::new();
    csv.push_str(&format!("n,m,time (ms),problemsize,adl compile time, bin compile time\n"));
    for m in measurements {
        csv.push_str(&format!("{},{},{},{},{},{}\n", m.0, m.1, m.2, m.3, m.4, m.5));
    }
    fs::write("tests/benchmarks/reachability/results.csv", csv).expect("Unable to write csv file.");
}

fn time_binary(file : &str, n : u32) -> (f64, f64, f64) {
    let mut avg = 0.0;
    let mut min_time : f64 = f64::MAX;
    let mut max_time : f64 = 0.0;

    println!("000/{:03}", n);
    for i in 0..n {
        let out = std::process::Command::new(file)
        .output()
        .expect(&format!("binary {} would not start.", file));
        let time_str = str::from_utf8(&out.stderr).unwrap();
        let time = time_str[..time_str.len()-4].parse::<f64>().unwrap();
        avg += time;
        min_time = f64::min(min_time, time);
        max_time = f64::max(max_time, time);

        println!("{:03}/{:03}", i+1, n);
    }
    return (((avg / (n as f64))*100.0).round()/100.0, min_time, max_time);
}

fn compile_adl_to_cuda(file_in: &str, file_out: &str, nrof_structs: u32) -> u64 {
    let now = Instant::now();
    let _ = std::process::Command::new("target/release/adl")
        .arg("-o")
        .arg(file_out)
        .arg("-n")
        .arg(nrof_structs.to_string())
        .arg(file_in)
        .output()
        .expect("adl could not compile");
    let elapsed_time = now.elapsed();
    return elapsed_time.as_secs();
}

fn compile_cuda_to_bin(file_in: &str, file_out: &str) -> u64 {
    let now = Instant::now();
    let _ = std::process::Command::new("nvcc")
        .arg("-o")
        .arg(file_out)
        .arg(file_in)
        .output()
        .expect("nvcc would not start.");
    let elapsed_time = now.elapsed();
    return elapsed_time.as_secs();
}