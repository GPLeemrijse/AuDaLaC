use crate::frontend::ast::Schedule;

pub fn fixpoint_depth(schedule: &Schedule) -> usize {
    use crate::frontend::ast::Schedule::*;
    match schedule {
        Sequential(s1, s2, _) => std::cmp::max(fixpoint_depth(s1), fixpoint_depth(s2)),
        Fixpoint(s, _) => fixpoint_depth(s) + 1,
        _ => 0,
    }
}

/*  Unfolds Sequential steps and returns the first non-Sequential (sub)schedule.
    Does return itself if non-Sequential.
*/
pub fn earliest_subschedule(schedule: &Schedule) -> &Schedule {
    use crate::frontend::ast::Schedule::*;
    match schedule {
        Sequential(s, _, _) => earliest_subschedule(s),
        StepCall(..) | TypedStepCall(..) | Fixpoint(..) => &schedule,
    }
}

pub fn step_calls<'a>(schedule: &'a Schedule, result: &mut Vec<(Option<&'a String>, &'a String)>) {
    use crate::frontend::ast::Schedule::*;

    match schedule {
        StepCall(step, _) => result.push((None, step)),
        TypedStepCall(strct, step, _) => result.push((Some(strct), step)),
        Sequential(s1, s2, _) => {
            step_calls(s1, result);
            step_calls(s2, result);
        }
        Fixpoint(s, _) => step_calls(s, result),
    }
}