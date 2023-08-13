mod in_kernel_alternating;
mod in_kernel_simple;
mod on_host_simple;
mod on_host_alternating;
mod graph_shared;
mod graph_shared_banks;
mod graph_shared_opportunistic;
mod graph_simple;

pub use in_kernel_alternating::InKernelAlternatingFixpoint;
pub use in_kernel_simple::InKernelSimpleFixpoint;
pub use on_host_simple::OnHostSimpleFixpoint;
pub use on_host_alternating::OnHostAlternatingFixpoint;
pub use graph_shared::GraphSharedFixpoint;
pub use graph_shared_banks::GraphSharedBanksFixpoint;
pub use graph_shared_opportunistic::GraphSharedBanksOpportunisticFixpoint;
pub use graph_simple::GraphSimpleFixpoint;