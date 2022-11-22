use assert_cmd::Command;
use std::fs;
use std::str;

#[test]
fn test_reachability_benchmarks() {
    let paths : Vec<String> = fs::read_dir("tests/benchmarks/reachability").unwrap().map(|p| p.as_ref().unwrap().path().display().to_string()).collect();
    let mut measurements : Vec<(i32, i32, f64)> = Vec::new();
    let nrof_files = paths.len();

    for (idx, path) in paths.iter().enumerate() {
        if path.ends_with(".adl") {
            let last_slash = path.rfind('/').unwrap();
            let file_name = &path[last_slash+1..path.len()-4];
            let cu_file = format!("/tmp/{file_name}.cu");
            let bin_file = format!("/tmp/{file_name}.out");
            let parts: Vec<&str> = file_name.split("_").collect();
            let n = parts[parts.len()-2].parse::<u32>().unwrap();
            let m = parts[parts.len()-1].parse::<u32>().unwrap();

            println!("{}/{}: Compiling {} to CUDA...", idx + 1, nrof_files, file_name);
            compile_adl_to_cuda(&path, &cu_file, n*m + 10);
            println!("Compiling {} to binary...", file_name);
            compile_cuda_to_bin(&cu_file, &bin_file);
            print!("Running {} on GPU.", file_name);
            let time = time_binary(&bin_file);
            println!("Completed in {} ms\n", time);
            
            measurements.push((n as i32, m as i32, time));
        }
    }

    let mut csv = String::new();
    csv.push_str(&format!("n,m,time (ms)\n"));
    for m in measurements {
        csv.push_str(&format!("{},{},{}\n", m.0, m.1, m.2));
    }
    fs::write("tests/benchmarks/reachability/results.csv", csv).expect("Unable to write csv file.");
}

fn time_binary(file : &str) -> f64 {
    let mut avg = 0.0;
    let n = 10.0;

    for _ in 0..(n as i64) {
        let out = std::process::Command::new(file)
        .output()
        .expect("binary would not start.");
        let time_str = str::from_utf8(&out.stderr).unwrap();
        avg += time_str[..time_str.len()-3].parse::<f64>().unwrap();
        print!(".");
    }
    println!("");
    return ((avg / n)*100.0).round()/100.0;
}

fn compile_adl_to_cuda(file_in: &str, file_out: &str, nrof_structs: u32) {
    let _ = Command::cargo_bin("adl")
        .unwrap()
        .arg("-o")
        .arg(file_out)
        .arg("-n")
        .arg(nrof_structs.to_string())
        .arg(file_in)
        .assert()
        .success();
}

fn compile_cuda_to_bin(file_in: &str, file_out: &str) {
    let _ = std::process::Command::new("nvcc")
        .arg("-o")
        .arg(file_out)
        .arg(file_in)
        .output()
        .expect("nvcc would not start.");
}