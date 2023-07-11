use crate::parser::ast::*;


pub fn visit_step<'a, T, I1, I2>(
    step: &'a Step,
    col: &mut T,
    f_stmt: fn(&'a Stat, &mut T, &Option<I1>),
    i_stmt: &Option<I1>,
    f_exp: fn(&'a Exp, &mut T, &Option<I2>),
    i_exp: &Option<I2>,
) {

    for s in &step.statements {
        visit_stat(s, col, f_stmt, i_stmt, f_exp, i_exp);
    }
}

pub fn visit_stat<'a, T, I1, I2>(
    stat: &'a Stat,
    col: &mut T,
    f_stmt: fn(&'a Stat, &mut T, &Option<I1>),
    i_stmt: &Option<I1>,
    f_exp: fn(&'a Exp, &mut T, &Option<I2>),
    i_exp: &Option<I2>,
) {
    use Stat::*;
    f_stmt(stat, col, i_stmt);

    match stat {
        IfThen(cond, stmts1, stmts2, _) => {
            visit_exp(cond, col, f_exp, i_exp);
            stmts1.iter()
                  .chain(
                    stmts2.iter()
                  )
                  .for_each(
                    |s| visit_stat(s, col, f_stmt, i_stmt, f_exp, i_exp)
                  );
        }
        Assignment(lhs, rhs, _) => {
            visit_exp(lhs, col, f_exp, i_exp);
            visit_exp(rhs, col, f_exp, i_exp);
        }
        Declaration(_, _, e, _) => {
            visit_exp(e, col, f_exp, i_exp);
        }
        _ => (),
    }
}

pub fn visit_exp<'a, T, I>(
    exp: &'a Exp,
    col: &mut T,
    f_exp: fn(&'a Exp, &mut T, &Option<I>),
    i_exp: &Option<I>,
) {
    use Exp::*;
    f_exp(exp, col, i_exp);

    match exp {
        BinOp(e1, _, e2, _) => {
            f_exp(e1, col, i_exp);
            f_exp(e2, col, i_exp);
        }
        UnOp(_, e, _) => {
            f_exp(e, col, i_exp);
        }
        Constructor(_, exps, _) => {
            for e in exps{
                f_exp(e, col, i_exp);
            }
        }
        _ => (),
    }
}