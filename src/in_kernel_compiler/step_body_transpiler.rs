use crate::MemOrder;
use crate::coalesced_compiler::as_c_literal;
use crate::in_kernel_compiler::as_c_type;
use crate::ast::*;
use std::collections::HashMap;

pub struct StepBodyTranspiler<'a> {
	var_exp_type_info : &'a HashMap<*const Exp, Vec<Type>>,
	memorder : MemOrder,
}


impl StepBodyTranspiler<'_> {
	pub fn new(type_info : &HashMap<*const Exp, Vec<Type>>, order: MemOrder) -> StepBodyTranspiler {
		StepBodyTranspiler{
			var_exp_type_info : type_info,
			memorder : order
		}
	}


	fn statements_as_c(&self, statements: &Vec<Stat>, strct: &ADLStruct, step: &Step, indent_lvl: usize) -> String {
		let indent = String::from("\t").repeat(indent_lvl);
		let mut stmt_vec : Vec<String> = Vec::new();

		for stmt in statements {
			use crate::ast::Stat::*;

			let statement_as_string = match stmt {
				IfThen(e, stmts_true, stmts_false, _) => {
					let cond = self.expression_as_c(e, strct, step);
					let body_true = self.statements_as_c(stmts_true, strct, step, indent_lvl + 1);
					let body_false = self.statements_as_c(stmts_false, strct, step, indent_lvl + 1);
					
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
				Assignment(parts_exp, e, _) => {
					let parts = parts_exp.get_parts();
					debug_assert!(parts.len() > 0);

					let types = self.var_exp_type_info.get(&(parts_exp as *const Exp)).expect("Could not find var expression in type info.");
					let e_as_c = self.expression_as_c(e, strct, step);
					
					let parts_as_c = self.expression_as_c(
						&Exp::Var(
							parts.to_vec(),
							(0, 0)
						),
						strct,
						step
					);

					let owner = self.expression_as_c(
						&Exp::Var(
							parts[..parts.len()-1].to_vec(),
							(0, 0)
						),
						strct,
						step
					);

					if strct.parameter_by_name(&parts[0]).is_some() {
						let param_type = self.basic_type_as_c(types.last().unwrap());
						let param = parts.last().unwrap();
						let owner_type = if parts.len() == 1 {
							strct.name.to_lowercase()
						} else {
							types[types.len()-2].name().unwrap().to_string().to_lowercase()
						};
						let load_suffix = self.load_suffix();
						let store_suffix = self.store_suffix("new_val");
						let adl_parts = parts.join(".");

						formatdoc!{"
							{indent}/* {adl_parts} = {e_as_c} */
							{indent}par_owner = {owner};
							{indent}if(par_owner != 0){{
							{indent}	{param_type} prev_val = {owner_type}->{param}[par_owner]{load_suffix};
							{indent}	{param_type} new_val = {e_as_c};
							{indent}	if (prev_val != new_val) {{
							{indent}		{owner_type}->{param}[par_owner]{store_suffix};
							{indent}		FP->set();
							{indent}	}}
							{indent}}}
						"}
					} else {
						format!("{indent}{parts_as_c} = {e_as_c};")
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
	    use std::iter::zip;
	    
	    match e {
	        BinOp(e1, c, e2, _) => {
	            let e1_comp = self.expression_as_c(e1, strct, step);
	            let e2_comp = self.expression_as_c(e2, strct, step);

	            format!("({e1_comp} {c} {e2_comp})")
	        },
	        UnOp(c, e, _) => {
	            let e_comp = self.expression_as_c(e, strct, step);
	            format!("({c}{e_comp})")
	        },
	        Constructor(n, args, _) => {
	            let args_comp = args.iter()
	                    .map(|e| self.expression_as_c(e, strct, step))
	                    .reduce(|acc: String, nxt| acc + ", " + &nxt).unwrap();

	            format!("{}->create_instance({args_comp})", n.to_lowercase())
	        },
	        Var(parts, _) => {
	        	if parts.is_empty() {
	        		// Used for statements_as_c
	        		return "self".to_string();
	        	}
	        	let is_parameter = strct.parameter_by_name(&parts[0]).is_some();
	        	let types = self.var_exp_type_info.get(&(e as *const Exp)).expect("Could not find var expression in type info.");

	        	let load_suffix = self.load_suffix();

	        	let mut previous_c_expr : String = "self".to_string();
	        	let mut previous_c_type : &String = &strct.name;
	        	let mut start_idx = 0;
	        	
	        	if !is_parameter {
	        		previous_c_expr = parts[0].clone();
	        		if let Some(t_name) = types[0].name() {
	        			previous_c_type = t_name;
	        		} else {
	        			debug_assert!(parts.len() == 1);
	        		}
	        		start_idx = 1;
	        	}

	        	for (t, p) in zip(types[start_idx..].iter(), parts[start_idx..].iter()) {
	        		previous_c_expr = format!("{}->{p}[{previous_c_expr}]{load_suffix}", previous_c_type.to_lowercase());
	        		if t.name().is_some() {
	        			previous_c_type = t.name().unwrap();
	        		}
	        	}

	            return format!("({previous_c_expr})");
	        },
	        Lit(l, _) => as_c_literal(l),
	    }
	}

	fn load_suffix(&self) -> String {
		if self.memorder.is_strong() {
			let order = self.memorder.as_cuda_order();
			format!(".load({order})")
		} else {
			"".to_string()
		}
	}

	fn store_suffix(&self, val : &str) -> String {
		if self.memorder.is_strong() {
			let order = self.memorder.as_cuda_order();
			format!(".store({val}, {order})")
		} else {
			format!(" = {val}")
		}
	}
}