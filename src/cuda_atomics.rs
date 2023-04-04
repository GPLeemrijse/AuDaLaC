pub enum MemOrder {
    Weak,
    Relaxed,
    SeqCons
}

impl MemOrder {
    pub fn from_str(s : &str) -> MemOrder {
        match s {
            "weak" => MemOrder::Weak,
            "relaxed" => MemOrder::Relaxed,
            "seqcons" => MemOrder::SeqCons,
            _ => panic!("Invalid MemOrder"),
        }
    }

    pub fn is_strong(&self) -> bool {
        !matches!(self, MemOrder::Weak)
    }

    pub fn as_cuda_order(&self) -> String {
        match self {
            MemOrder::Weak => panic!("Weak order should not be translated to c."),
            MemOrder::Relaxed => "cuda::memory_order_relaxed".to_string(),
            MemOrder::SeqCons => "cuda::memory_order_seq_cst".to_string(),
        }
    }
}

pub enum Scope {
    System,
    Device
}

impl Scope {
    pub fn from_str(s : &str) -> Scope {
        match s {
            "system" => Scope::System,
            "device" => Scope::Device,
            _ => panic!("Invalid scope"),
        }
    }

    pub fn as_cuda_scope(&self) -> String {
        (match self {
            Scope::System => "cuda::thread_scope_system",
            Scope::Device => "cuda::thread_scope_device"
        }).to_string()
    }
}