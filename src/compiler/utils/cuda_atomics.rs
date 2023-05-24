pub enum MemOrder {
    Weak,
    Relaxed,
    AcqRel,
    SeqCons,
}

pub enum MemoryOperation {
    Load,
    Store,
}

impl MemOrder {
    pub fn from_str(s: &str) -> MemOrder {
        match s {
            "weak" => MemOrder::Weak,
            "relaxed" => MemOrder::Relaxed,
            "seqcons" => MemOrder::SeqCons,
            "acqrel" => MemOrder::AcqRel,
            _ => panic!("Invalid MemOrder"),
        }
    }

    pub fn is_strong(&self) -> bool {
        !matches!(self, MemOrder::Weak)
    }

    pub fn as_cuda_order(&self, op: Option<MemoryOperation>) -> String {
        match (self, op) {
            (MemOrder::Weak, _) => panic!("Weak order should not be translated to c."),
            (MemOrder::Relaxed, _) => "cuda::memory_order_relaxed".to_string(),
            (MemOrder::SeqCons, _) => "cuda::memory_order_seq_cst".to_string(),
            (MemOrder::AcqRel, Some(MemoryOperation::Load)) => {
                "cuda::memory_order_acquire".to_string()
            }
            (MemOrder::AcqRel, Some(MemoryOperation::Store)) => {
                "cuda::memory_order_release".to_string()
            }
            _ => panic!("Unsupported combination of memory order and operation."),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Scope {
    System,
    Device,
}

impl Scope {
    pub fn from_str(s: &str) -> Scope {
        match s {
            "system" => Scope::System,
            "device" => Scope::Device,
            _ => panic!("Invalid scope"),
        }
    }

    pub fn as_cuda_scope(&self) -> String {
        (match self {
            Scope::System => "cuda::thread_scope_system",
            Scope::Device => "cuda::thread_scope_device",
        })
        .to_string()
    }
}
