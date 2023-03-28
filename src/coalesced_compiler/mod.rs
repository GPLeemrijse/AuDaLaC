mod coalesced_transpiler;
mod coalesced_struct_manager;
mod coalesced_schedule_manager;
mod coalesced_ast_translator;

pub use coalesced_transpiler::CoalescedCUDATranspiler; 
pub use coalesced_struct_manager::CoalescedStructManager;
pub use coalesced_schedule_manager::CoalescedScheduleManager;
pub use coalesced_ast_translator::{as_c_type, as_type_enum, as_printf, as_c_literal};