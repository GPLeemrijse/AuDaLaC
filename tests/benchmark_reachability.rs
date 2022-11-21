use assert_cmd::Command;
use std::fs;
use std::str;

#[test]
fn test_reachability_benchmarks() {
    let paths = fs::read_dir("tests/benchmarks/reachability").unwrap();
    let mut measurements : Vec<(i32, i32, f64)> = Vec::new();
    for path in paths {
        let path_string = path.as_ref().unwrap().path().display().to_string();
        if path_string.ends_with(".adl") {
            let last_slash = path_string.rfind('/').unwrap();
            let file_name = &path_string[last_slash+1..path_string.len()-4];
            let cu_file = format!("/tmp/{file_name}.cu");
            let bin_file = format!("/tmp/{file_name}.out");
            println!("Compiling {} to CUDA...", file_name);
            compile_adl_to_cuda(&path_string, &cu_file);
            println!("Compiling {} to binary...", file_name);
            compile_cuda_to_bin(&cu_file, &bin_file);
            println!("Running {} on GPU...", file_name);
            let time_str = time_binary(&bin_file);
            let time = time_str[..time_str.len()-3].parse::<f64>().unwrap();
            let parts: Vec<&str> = file_name.split("_").collect();
            measurements.push((parts[parts.len()-2].parse::<i32>().unwrap(), parts[parts.len()-1].parse::<i32>().unwrap(), time));
        }
    }

    let mut csv = String::new();
    csv.push_str(&format!("file_name,n,m\n"));
    for m in measurements {
        csv.push_str(&format!("{},{},{}\n", m.0, m.1, m.2));
    }
    fs::write("tests/benchmarks/reachability/results.csv", csv).expect("Unable to write csv file.");
}

fn time_binary(file : &str) -> String {
    let out = std::process::Command::new(file)
        .output()
        .expect("binary would not start.");
    return str::from_utf8(&out.stderr).unwrap().to_string();
}

fn compile_adl_to_cuda(file_in: &str, file_out: &str) {
    let out = Command::cargo_bin("adl")
        .unwrap()
        .arg("-o")
        .arg(file_out)
        .arg(file_in)
        .assert()
        .success();
}

fn compile_cuda_to_bin(file_in: &str, file_out: &str) {
    let out = std::process::Command::new("nvcc")
        .arg("-o")
        .arg(file_out)
        .arg(file_in)
        .output()
        .expect("nvcc would not start.");
}