use crate::transpilation_traits::FPStrategy;
use crate::MemOrder;
use crate::coalesced_compiler::as_c_literal;
use crate::in_kernel_compiler::as_c_type;
use crate::ast::*;
use std::collections::HashMap;
use indoc::formatdoc;

pub struct StepBodyTranspiler<'a> {
	var_exp_type_info : &'a HashMap<*const Exp, Vec<Type>>,
	memorder : &'a MemOrder,
	fp : &'a dyn FPStrategy,
}


impl StepBodyTranspiler<'_> {
	pub fn new<'a>(type_info : &'a HashMap<*const Exp, Vec<Type>>, order: &'a MemOrder, fp : &'a dyn FPStrategy) -> StepBodyTranspiler<'a> {
		StepBodyTranspiler{
			var_exp_type_info : type_info,
			memorder : order,
			fp
		}
	}


	pub fn statements_as_c(&self, statements: &Vec<Stat>, strct: &ADLStruct, step: &Step, indent_lvl: usize, fp_level : usize) -> String {
		let indent = String::from("\t").repeat(indent_lvl);
		let mut stmt_vec : Vec<String> = Vec::new();

		for stmt in statements {
			use crate::ast::Stat::*;

			let statement_as_string = match stmt {
				IfThen(e, stmts_true, stmts_false, _) => {
					let cond = self.expression_as_c(e, strct, step);
					let body_true = self.statements_as_c(stmts_true, strct, step, indent_lvl + 1, fp_level);
					let body_false = self.statements_as_c(stmts_false, strct, step, indent_lvl + 1, fp_level);
					
					let else_block = if body_false == "" { "".to_string()} else {
						format!(" else {{{body_false}\n{indent}}}")
					};

					format!{"{indent}if {cond} {{{body_true}\n{indent}}}{else_block}"}
				},
				Declaration(t, n, e, _) => {
					let t_as_c = as_c_type(t);
					let e_as_c = self.expression_as_c(e, strct, step);
					format!("{indent}{t_as_c} {n} = {e_as_c};")
				},
				Assignment(lhs_exp, rhs_exp, _) => {
					let (lhs_as_c, owner, is_parameter) = self.var_exp_as_c(lhs_exp, strct);
					let rhs_as_c = self.expression_as_c(rhs_exp, strct, step);

					if is_parameter {
						let (owner_exp, owner_type) = owner.expect("Expected an owner for a parameter assignment.");
						let parts_vec = lhs_exp.get_parts();

						let adl_parts = parts_vec.join(".");
						let param = parts_vec.last().unwrap();

						let set_fp = self.fp.set_unstable(fp_level);

						formatdoc!{"
							{indent}/* {adl_parts} = {rhs_as_c} */
							{indent}owner = {owner_exp};
							{indent}if(owner != 0){{
							{indent}	auto prev_val = LOAD({owner_type}->{param}[owner]);
							{indent}	auto new_val = {rhs_as_c};
							{indent}	if (prev_val != new_val) {{
							{indent}		STORE({owner_type}->{param}[owner], new_val);
							{indent}		{set_fp}
							{indent}	}}
							{indent}}}
						"}
					} else {
						format!("{indent}{lhs_as_c} = {rhs_as_c};")
					}
				},
			};
			stmt_vec.push(statement_as_string);
		}

		return stmt_vec.iter()
					   .fold("".to_string(), |acc: String, nxt| acc + "\n" + nxt);
	}

	fn expression_as_c(&self, e: &Exp, strct: &ADLStruct, step: &Step) -> String {
	    use Exp::*;
	    
	    match e {
	        BinOp(e1, c, e2, _) => {
	            let e1_comp = self.expression_as_c(e1, strct, step);
	            let e2_comp = self.expression_as_c(e2, strct, step);

	            format!("({e1_comp} {c} {e2_comp})")
	        },
	        UnOp(c, e, _) => {
	            let e_comp = self.expression_as_c(e, strct, step);
	            format!("{c}{e_comp}")
	        },
	        Constructor(n, args, _) => {
	            let args_comp = args.iter()
	                    .map(|e| self.expression_as_c(e, strct, step))
	                    .reduce(|acc: String, nxt| acc + ", " + &nxt).unwrap();

	            format!("{}->create_instance({args_comp})", n.to_lowercase())
	        },
	        Var(..) => {
	        	self.var_exp_as_c(e, strct).0 // Not interested in the owner or is_parameter
	        },
	        Lit(l, _) => as_c_literal(l),
	    }
	}

	fn var_exp_as_c(&self, exp : &Exp, strct: &ADLStruct) -> (String, Option<(String, Type)>, bool) {
		use std::iter::zip;
		if let Exp::Var(parts, _) = exp {
			let types = self.var_exp_type_info.get(&(exp as *const Exp)).expect("Could not find var expression in type info.");
			let is_parameter = strct.parameter_by_name(&parts[0]).is_some();

			let (exp_as_c, owner);

        	// For parameters, the 'self' part is implicit.
        	if is_parameter {
        		(exp_as_c, owner) = self.stitch_parts_and_types(zip(
        			["self".to_string()].iter().chain(parts.iter()),
        			[Type::Named(strct.name.clone())].iter().chain(types.iter())
        		));
        	} else {
        		(exp_as_c, owner) = self.stitch_parts_and_types(zip(
        			parts.iter(),
        			types.iter()
        		));
        	}
        	return (exp_as_c, owner, is_parameter);
		}
		panic!("Expected Var expression.");
	}

	/* Returns the full expression for evaluating the field-type pairs in parts, 
	   and, if applicable, returns the penultimate partial result (the parameter owner).
	*/
	fn stitch_parts_and_types<'a, I>(&self, mut parts : I) -> (String, Option<(String, Type)>)
	where
		I : Iterator<Item = (&'a String, &'a Type)>
	{
		let (p0, mut previous_c_type) = parts.next().expect("Supply at least one part-type pair to get_var_expr_as_c.");
		let mut previous_c_expr = p0.clone();
		let mut owner = None;

		let mut peekable = parts.peekable();

		while let Some((id, id_type)) = peekable.next() {
			// non-named types are only allowed at the end. validate_ast should catch this.
			if previous_c_type.name().is_none() && peekable.peek().is_some() {
				panic!("non-named type not at the end of var expression.");
			}

			// Store the penultimate value in owner
			if peekable.peek().is_none() {
				owner = Some((previous_c_expr.clone(), previous_c_type.clone()));
			}
			
			previous_c_expr = format!("LOAD({}->{id}[{previous_c_expr}])", previous_c_type.name().unwrap().to_lowercase());
			previous_c_type = id_type;
		}

		(previous_c_expr, owner)
	}
}