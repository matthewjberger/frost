use std::collections::HashMap;

use anyhow::{Result, bail};
use cranelift::codegen::ir::{ArgumentPurpose, StackSlot};
use cranelift::prelude::*;
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::c_abi::{CArgument, CLayout, CRegister, CReturn};
use crate::ir::{
    IrBinOp, IrConstant, IrFunction, IrModule, IrOperand, IrRvalue,
    IrStatement, IrTerminator, IrUnOp,
};
use crate::types::Type;

// The runtime's index check. Every checked read names it, and this backend
// answers for it itself rather than calling it, so it is written once here.
const BOUNDS_CHECK: &str = "frost_rt_bounds_check";

// `FROST_TIMINGS=1` reports the split between generating code for each
// function and writing the object, which are the two halves of the backend and
// want different answers. One parallelizes, the other wants more compilation
// units. Off unless asked for, and it prints to stderr so it never reaches
// emitted output.
// One ISA per thread. Sharing a single one across threads is allowed, but the
// scaling says something behind it serializes, and an ISA is cheap to build
// next to a second of code generation.
fn make_isa() -> Result<std::sync::Arc<dyn cranelift::codegen::isa::TargetIsa>>
{
    let mut flag_builder = settings::builder();
    flag_builder.set("opt_level", "speed")?;
    flag_builder.set("is_pic", "true")?;
    // Windows grows a thread's stack through a guard page, so a frame wider
    // than one page has to touch each page on the way down or the guard is
    // stepped over and the write faults. Without this a function holding a
    // couple of kilobytes of locals crashes on the first write.
    flag_builder.set("enable_probestack", "true")?;
    flag_builder.set("probestack_strategy", "inline")?;
    let isa_builder = cranelift_native::builder()
        .map_err(|message| anyhow::anyhow!("ISA builder: {message}"))?;
    Ok(isa_builder.finish(settings::Flags::new(flag_builder))?)
}

fn timings_wanted() -> bool {
    std::env::var("FROST_TIMINGS").is_ok_and(|value| value != "0")
}

pub fn compile_ir_to_object(module: &IrModule) -> Result<Vec<u8>> {
    let report = timings_wanted();
    let started = std::time::Instant::now();
    let (mut object, mut generator) = Generator::new()?;
    generator.declare_strings(&mut object, module)?;
    generator.declare_functions(&mut object, module)?;
    let declared = started.elapsed();

    // Code generation is nearly all of the backend's time, it is per function,
    // and the inputs are independent once every symbol is declared. So it runs
    // on as many threads as the machine has, and only the defining is serial,
    // because the object is one mutable thing.
    let generator = &generator;
    let isa = object.isa();
    // `FROST_THREADS` caps it, both so a build system can leave headroom and so
    // the scaling can be swept rather than guessed at.
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let requested = std::env::var("FROST_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|count| *count > 0)
        .unwrap_or(available);
    let threads = requested.min(module.functions.len().max(1));
    let mut compiled: Vec<Compiled> =
        Vec::with_capacity(module.functions.len());

    if threads <= 1 {
        let mut context = cranelift::codegen::Context::new();
        let mut builder_context = FunctionBuilderContext::new();
        for function in &module.functions {
            compiled.push(compile_one(
                generator,
                isa,
                function,
                &mut context,
                &mut builder_context,
            )?);
        }
    } else {
        // Functions are handed out one at a time from a shared cursor rather
        // than split into equal chunks up front. Cost per function varies by
        // more than an order of magnitude and the expensive ones sit next to
        // each other in the module, so any static split leaves one thread
        // holding all of them while the rest finish early and wait.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let next = &next;
        let results: Vec<Result<Vec<(usize, Compiled)>>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|_| {
                        scope.spawn(move || {
                            let isa = make_isa()?;
                            let isa = isa.as_ref();
                            let mut context =
                                cranelift::codegen::Context::new();
                            let mut builder_context =
                                FunctionBuilderContext::new();
                            let mut out = Vec::new();
                            loop {
                                let index = next.fetch_add(
                                    1,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                let Some(function) =
                                    module.functions.get(index)
                                else {
                                    break;
                                };
                                out.push((
                                    index,
                                    compile_one(
                                        generator,
                                        isa,
                                        function,
                                        &mut context,
                                        &mut builder_context,
                                    )?,
                                ));
                            }
                            Ok(out)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| match handle.join() {
                        Ok(result) => result,
                        Err(_) => bail!(
                            "native backend: a code generation thread panicked"
                        ),
                    })
                    .collect()
            });
        // Sorted back into module order, so the object a build produces does
        // not depend on how the threads happened to interleave.
        let mut ordered: Vec<(usize, Compiled)> = Vec::new();
        for result in results {
            ordered.extend(result?);
        }
        ordered.sort_by_key(|(index, _)| *index);
        compiled.extend(ordered.into_iter().map(|(_, unit)| unit));
    }

    let generated = started.elapsed();
    for unit in &compiled {
        object.define_function_bytes(
            unit.id,
            &unit.function,
            unit.alignment as u64,
            &unit.bytes,
            &unit.relocations,
        )?;
    }
    let defined = started.elapsed();
    let product = object.finish();
    let bytes = product.emit()?;
    if report {
        eprintln!(
            "frost: {} functions on {threads} thread(s), declaring {:.0} ms, code generation {:.0} ms, defining {:.0} ms, object emission {:.0} ms",
            module.functions.len(),
            declared.as_secs_f64() * 1000.0,
            (generated - declared).as_secs_f64() * 1000.0,
            (defined - generated).as_secs_f64() * 1000.0,
            (started.elapsed() - defined).as_secs_f64() * 1000.0
        );
    }
    Ok(bytes)
}

// One function's finished machine code, held by value so nothing borrows the
// context it was compiled in once the thread that owned it is gone.
struct Compiled {
    id: FuncId,
    // Needed only so the object backend can resolve relocation targets against
    // the function's imported names. Moved out of the context rather than
    // cloned, since the context is cleared before it is used again.
    function: cranelift::codegen::ir::Function,
    alignment: u32,
    bytes: Vec<u8>,
    relocations: Vec<cranelift::codegen::FinalizedMachReloc>,
}

fn compile_one(
    generator: &Generator,
    isa: &dyn cranelift::codegen::isa::TargetIsa,
    function: &IrFunction,
    context: &mut cranelift::codegen::Context,
    builder_context: &mut FunctionBuilderContext,
) -> Result<Compiled> {
    context.clear();
    let id = generator.build_function(function, context, builder_context)?;
    let (alignment, bytes, relocations) = {
        let code = context
            .compile(
                isa,
                &mut cranelift::codegen::control::ControlPlane::default(),
            )
            .map_err(|error| {
                anyhow::anyhow!("native backend: {:?}", error.inner)
            })?;
        (
            code.buffer.alignment,
            code.buffer.data().to_vec(),
            code.buffer.relocs().to_vec(),
        )
    };
    Ok(Compiled {
        id,
        function: std::mem::replace(
            &mut context.func,
            cranelift::codegen::ir::Function::new(),
        ),
        alignment,
        bytes,
        relocations,
    })
}

// What building a function's body needs to know about the rest of the module:
// the call convention and, for anything it references, that symbol's signature
// and whether it is colocated. This is deliberately not the module. A module is
// a single mutable thing and code generation is the part of the backend worth
// doing on more than one thread at a time, so the translation is given the facts
// rather than the object that owns them.
//
// The two helpers replicate `Module::declare_func_in_func` and
// `declare_data_in_func` exactly. Both of those take `&mut self` but read only
// `self.declarations()`, which is what makes this safe to do from a snapshot.
struct Decls {
    call_conv: isa::CallConv,
    functions: HashMap<FuncId, (Signature, bool)>,
    data: HashMap<DataId, bool>,
    // How C returns each extern that returns an aggregate. Frost returns its
    // own aggregates through a hidden out-pointer no matter what they are. C
    // does not, so a call to one of these is emitted from what
    // `src/c_abi.rs` says the target does rather than from what Frost does.
    c_returns: HashMap<String, (CLayout, CReturn)>,
    // Extern name to the parameters it takes by value: which argument, its
    // layout, and how the target passes it.
    c_arguments: HashMap<String, Vec<(usize, CLayout, CArgument)>>,
}

impl Decls {
    fn make_signature(&self) -> Signature {
        Signature::new(self.call_conv)
    }

    fn declare_func_in_func(
        &self,
        id: FuncId,
        func: &mut cranelift::codegen::ir::Function,
    ) -> cranelift::codegen::ir::FuncRef {
        use cranelift::codegen::ir;
        let (signature, colocated) = &self.functions[&id];
        let signature = func.import_signature(signature.clone());
        let name = func.declare_imported_user_function(ir::UserExternalName {
            namespace: 0,
            index: id.as_u32(),
        });
        func.import_function(ir::ExtFuncData {
            name: ir::ExternalName::user(name),
            signature,
            colocated: *colocated,
        })
    }

    fn declare_data_in_func(
        &self,
        id: DataId,
        func: &mut cranelift::codegen::ir::Function,
    ) -> cranelift::codegen::ir::GlobalValue {
        use cranelift::codegen::ir;
        let colocated = self.data[&id];
        let name = func.declare_imported_user_function(ir::UserExternalName {
            namespace: 1,
            index: id.as_u32(),
        });
        func.create_global_value(ir::GlobalValueData::Symbol {
            name: ir::ExternalName::user(name),
            offset: ir::immediates::Imm64::new(0),
            colocated,
            tls: false,
        })
    }
}

struct Generator {
    decls: Decls,
    functions: HashMap<String, FuncId>,
    return_types: HashMap<String, Type>,
    strings: HashMap<String, DataId>,
    pointer_type: types::Type,
    // `None` on a target whose C struct-return rule is not written down here,
    // which makes an aggregate-returning extern an error rather than a guess.
    c_target: Option<crate::c_abi::CTarget>,
}

impl Generator {
    fn new() -> Result<(ObjectModule, Self)> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed")?;
        flag_builder.set("is_pic", "true")?;
        flag_builder.set("enable_probestack", "true")?;
        flag_builder.set("probestack_strategy", "inline")?;
        let isa_builder = cranelift_native::builder()
            .map_err(|message| anyhow::anyhow!("ISA builder: {message}"))?;
        let isa = isa_builder.finish(settings::Flags::new(flag_builder))?;
        let pointer_type = isa.pointer_type();
        let builder = ObjectBuilder::new(
            isa,
            "frost_module",
            cranelift_module::default_libcall_names(),
        )?;
        let module = ObjectModule::new(builder);
        let call_conv = module.isa().default_call_conv();
        let c_target = crate::c_abi::target_of(module.isa().triple());
        Ok((
            module,
            Generator {
                decls: Decls {
                    call_conv,
                    functions: HashMap::new(),
                    data: HashMap::new(),
                    c_returns: HashMap::new(),
                    c_arguments: HashMap::new(),
                },
                c_target,
                functions: HashMap::new(),
                return_types: HashMap::new(),
                strings: HashMap::new(),
                pointer_type,
            },
        ))
    }

    fn declare_strings(
        &mut self,
        object: &mut ObjectModule,
        module: &IrModule,
    ) -> Result<()> {
        let mut counter = 0;
        collect_strings(module, &mut |text| {
            if self.strings.contains_key(text) {
                return Ok(());
            }
            let name = format!(".str.{counter}");
            counter += 1;
            let data_id =
                object.declare_data(&name, Linkage::Local, false, false)?;
            let mut description = DataDescription::new();
            let mut bytes = text.as_bytes().to_vec();
            bytes.push(0);
            description.define(bytes.into_boxed_slice());
            object.define_data(data_id, &description)?;
            self.decls.data.insert(data_id, true);
            self.strings.insert(text.to_string(), data_id);
            Ok(())
        })
    }

    fn declare_functions(
        &mut self,
        object: &mut ObjectModule,
        module: &IrModule,
    ) -> Result<()> {
        let pointer_type = self.pointer_type;
        for external in &module.externs {
            let mut signature = self.decls.make_signature();
            // C returns an aggregate by its own rule, so the classification
            // decides the signature before the ordinary parameters are added:
            // an indirect return takes a hidden pointer as the first argument.
            let returned = match &external.return_layout {
                Some(layout) => {
                    let Some(target) = self.c_target else {
                        bail!(
                            "native backend: '{}' returns '{}' by value, and this target's C rule for returning a struct is not one Frost knows; see item 4 of docs/book/src/roadmap.md",
                            external.name,
                            external.return_type
                        );
                    };
                    let returned =
                        crate::c_abi::classify_return(layout, target);
                    if matches!(returned, CReturn::Indirect) {
                        signature.params.push(AbiParam::special(
                            pointer_type,
                            ArgumentPurpose::StructReturn,
                        ));
                    }
                    self.decls.c_returns.insert(
                        external.name.clone(),
                        (layout.clone(), returned.clone()),
                    );
                    Some(returned)
                }
                None => None,
            };
            // An aggregate parameter is a pointer by design:
            // `close :: extern fn(f: File)` links against `void close(File*)`.
            // That is the documented convention in docs/book/src/impl/c-compatibility.md.
            // A parameter written `value` is the exception, and it is the one
            // that has to be classified the way a return is, because C splits a
            // struct across registers by a rule of the target's.
            let mut by_value = Vec::new();
            for (index, parameter) in external.params.iter().enumerate() {
                let layout =
                    external.param_layouts.get(index).and_then(Clone::clone);
                let Some(layout) = layout else {
                    signature.params.push(AbiParam::new(param_abi_type(
                        pointer_type,
                        parameter,
                    )?));
                    continue;
                };
                let Some(target) = self.c_target else {
                    bail!(
                        "native backend: '{}' takes a parameter by value, and this target's C rule for passing a struct is not one Frost knows; see docs/book/src/impl/c-compatibility.md",
                        external.name
                    );
                };
                let passed = crate::c_abi::classify_argument(&layout, target);
                match &passed {
                    CArgument::Registers(registers) => {
                        for register in registers {
                            signature
                                .params
                                .push(AbiParam::new(register_type(*register)));
                        }
                    }
                    CArgument::Indirect => {
                        signature.params.push(AbiParam::new(pointer_type));
                    }
                    // A System V struct too large for registers goes on the
                    // stack as part of the argument area. Cranelift places it
                    // given the size, and the value handed over at the call is
                    // a pointer to the bytes, which it copies. The size is
                    // rounded up to a whole number of eightbytes because that
                    // is how the argument area is laid out and what the x64
                    // backend asserts.
                    CArgument::Stack => {
                        signature.params.push(AbiParam::special(
                            pointer_type,
                            ArgumentPurpose::StructArgument(
                                layout.size.next_multiple_of(8) as u32,
                            ),
                        ));
                    }
                }
                by_value.push((index, layout, passed));
            }
            if !by_value.is_empty() {
                self.decls
                    .c_arguments
                    .insert(external.name.clone(), by_value);
            }
            match &returned {
                Some(CReturn::Registers(registers)) => {
                    for register in registers {
                        signature
                            .returns
                            .push(AbiParam::new(register_type(*register)));
                    }
                }
                Some(CReturn::Indirect) => {}
                None => {
                    if !matches!(external.return_type, Type::Void) {
                        signature.returns.push(AbiParam::new(clif_type(
                            pointer_type,
                            &external.return_type,
                        )?));
                    }
                }
            }
            let func_id = object.declare_function(
                &external.name,
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions.insert(external.name.clone(), func_id);
            self.return_types
                .insert(external.name.clone(), external.return_type.clone());
        }

        for function in &module.functions {
            let signature = self.build_signature(function)?;
            // A specialization is private to the object that holds it, so two
            // objects that both instantiated the same generic do not collide.
            let linkage = if function.local {
                Linkage::Local
            } else {
                Linkage::Export
            };
            let func_id =
                object.declare_function(&function.name, linkage, &signature)?;
            self.decls.functions.insert(func_id, (signature, true));
            self.functions.insert(function.name.clone(), func_id);
            self.return_types
                .insert(function.name.clone(), function.return_type.clone());
        }

        // Functions another object defines. Declared with the same signature
        // builder as a definition, so the call this object emits agrees with
        // the definition the linker resolves it to.
        for function in &module.imported {
            if self.functions.contains_key(&function.name) {
                continue;
            }
            let signature = self.build_signature(function)?;
            let func_id = object.declare_function(
                &function.name,
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions.insert(function.name.clone(), func_id);
            self.return_types
                .insert(function.name.clone(), function.return_type.clone());
        }

        if !self.functions.contains_key("memcpy") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(pointer_type));
            signature.params.push(AbiParam::new(pointer_type));
            signature.params.push(AbiParam::new(pointer_type));
            signature.returns.push(AbiParam::new(pointer_type));
            let func_id = object.declare_function(
                "memcpy",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions.insert("memcpy".to_string(), func_id);
        }

        for name in [BOUNDS_CHECK, "frost_rt_generation_check"] {
            if self.functions.contains_key(name) {
                continue;
            }
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
            let func_id =
                object.declare_function(name, Linkage::Import, &signature)?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions.insert(name.to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_mem_set") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
            signature.params.push(AbiParam::new(types::I64));
            let func_id = object.declare_function(
                "frost_rt_mem_set",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_mem_set".to_string(), func_id);
        }

        // The pieces a formatted `print` is written as.
        if !self.functions.contains_key("frost_rt_write_bytes") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(pointer_type));
            signature.params.push(AbiParam::new(types::I64));
            let func_id = object.declare_function(
                "frost_rt_write_bytes",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_write_bytes".to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_write_i64") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(types::I64));
            let func_id = object.declare_function(
                "frost_rt_write_i64",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_write_i64".to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_write_f64") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(types::F64));
            let func_id = object.declare_function(
                "frost_rt_write_f64",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_write_f64".to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_write_cstr") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(pointer_type));
            let func_id = object.declare_function(
                "frost_rt_write_cstr",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_write_cstr".to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_write_newline") {
            let signature = self.decls.make_signature();
            let func_id = object.declare_function(
                "frost_rt_write_newline",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_write_newline".to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_print_i64") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(types::I64));
            let func_id = object.declare_function(
                "frost_rt_print_i64",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_print_i64".to_string(), func_id);
        }

        if !self.functions.contains_key("frost_rt_print_f64") {
            let mut signature = self.decls.make_signature();
            signature.params.push(AbiParam::new(types::F64));
            let func_id = object.declare_function(
                "frost_rt_print_f64",
                Linkage::Import,
                &signature,
            )?;
            self.decls.functions.insert(func_id, (signature, false));
            self.functions
                .insert("frost_rt_print_f64".to_string(), func_id);
        }

        Ok(())
    }

    fn function_return_type(&self, function: &IrFunction) -> Type {
        if function.name == "main" {
            Type::I32
        } else {
            function.return_type.clone()
        }
    }

    fn returns_aggregate(&self, function: &IrFunction) -> bool {
        function.name != "main" && is_aggregate(&function.return_type)
    }

    fn build_signature(
        &self,
        function: &IrFunction,
    ) -> Result<cranelift::codegen::ir::Signature> {
        let pointer_type = self.pointer_type;
        let mut signature = self.decls.make_signature();
        for index in 0..function.param_count {
            // A parameter C passes as the struct itself arrives the way that
            // target passes one: split across registers, or as an address the
            // caller already copied to. See build_function, which puts the
            // pieces back together.
            if let Some(layout) =
                function.param_layouts.get(index).and_then(Clone::clone)
            {
                let Some(target) = self.c_target else {
                    bail!(
                        "native backend: '{}' takes a parameter by value, and this target's C rule for passing a struct is not one Frost knows; see docs/book/src/impl/c-compatibility.md",
                        function.name
                    );
                };
                match crate::c_abi::classify_argument(&layout, target) {
                    CArgument::Registers(registers) => {
                        for register in registers {
                            signature
                                .params
                                .push(AbiParam::new(register_type(register)));
                        }
                    }
                    CArgument::Indirect => {
                        signature.params.push(AbiParam::new(pointer_type));
                    }
                    CArgument::Stack => {
                        signature.params.push(AbiParam::special(
                            pointer_type,
                            ArgumentPurpose::StructArgument(
                                layout.size.next_multiple_of(8) as u32,
                            ),
                        ));
                    }
                }
                continue;
            }
            signature.params.push(AbiParam::new(param_abi_type(
                pointer_type,
                function.local_type(index),
            )?));
        }
        if self.returns_aggregate(function) {
            signature.params.push(AbiParam::new(pointer_type));
        } else {
            let return_type = self.function_return_type(function);
            if !matches!(return_type, Type::Void) {
                signature.returns.push(AbiParam::new(clif_type(
                    pointer_type,
                    &return_type,
                )?));
            }
        }
        Ok(signature)
    }

    // Build a function's body into the given context, touching nothing shared.
    // This is the part that runs on every thread at once.
    fn build_function(
        &self,
        function: &IrFunction,
        context: &mut cranelift::codegen::Context,
        builder_context: &mut FunctionBuilderContext,
    ) -> Result<FuncId> {
        let func_id = self.functions[&function.name];
        let pointer_type = self.pointer_type;
        let returns_aggregate = self.returns_aggregate(function);
        let return_type = self.function_return_type(function);

        context.func.signature = self.build_signature(function)?;

        let mut builder =
            FunctionBuilder::new(&mut context.func, builder_context);

        let clif_blocks: Vec<Block> = function
            .blocks
            .iter()
            .map(|_| builder.create_block())
            .collect();
        let entry = clif_blocks[function.entry];
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);

        let mut slots: HashMap<usize, StackSlot> = HashMap::new();
        for (index, local) in function.locals.iter().enumerate() {
            if matches!(local.ty, Type::Void | Type::Unknown) {
                continue;
            }
            if local.in_memory {
                let size = local.size.max(1) as u32;
                let slot = builder.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    size,
                    0,
                ));
                slots.insert(index, slot);
            } else {
                builder.declare_var(
                    Variable::new(index),
                    clif_type(pointer_type, &local.ty)?,
                );
            }
        }

        let memcpy = self.functions["memcpy"];
        let params = builder.block_params(entry).to_vec();
        // A by-value parameter is more than one block parameter, so the two are
        // walked with a cursor rather than zipped.
        let mut at = 0usize;
        for index in 0..function.param_count {
            let local = &function.locals[index];
            if let Some(layout) =
                function.param_layouts.get(index).and_then(Clone::clone)
            {
                let target = self.c_target.expect("checked in build_signature");
                match crate::c_abi::classify_argument(&layout, target) {
                    // The pieces are put back into storage this function owns,
                    // and the parameter points at it, which is the shape the
                    // body already reads a borrowed struct through.
                    CArgument::Registers(registers) => {
                        let slot = builder.create_sized_stack_slot(
                            StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                layout.size.max(1) as u32,
                                4,
                            ),
                        );
                        let base =
                            builder.ins().stack_addr(pointer_type, slot, 0);
                        for register in &registers {
                            store_incoming_register(
                                &mut builder,
                                *register,
                                params[at],
                                base,
                                &layout,
                                pointer_type,
                                memcpy,
                                &self.decls,
                            );
                            at += 1;
                        }
                        builder.def_var(Variable::new(index), base);
                    }
                    // Both of these arrive as an address the caller already
                    // copied to, so the copy this function is entitled to is
                    // the one it was handed.
                    CArgument::Indirect | CArgument::Stack => {
                        builder.def_var(Variable::new(index), params[at]);
                        at += 1;
                    }
                }
                continue;
            }
            let value = params[at];
            at += 1;
            if is_aggregate(&local.ty) {
                let slot = slots[&index];
                let destination =
                    builder.ins().stack_addr(pointer_type, slot, 0);
                let size =
                    builder.ins().iconst(pointer_type, local.size as i64);
                let memcpy_ref =
                    self.decls.declare_func_in_func(memcpy, builder.func);
                builder.ins().call(memcpy_ref, &[destination, value, size]);
            } else if let Some(slot) = slots.get(&index) {
                builder.ins().stack_store(value, *slot, 0);
            } else {
                builder.def_var(Variable::new(index), value);
            }
        }
        let out_pointer = if returns_aggregate {
            Some(params[at])
        } else {
            None
        };

        {
            let mut translator = Translator {
                decls: &self.decls,
                functions: &self.functions,
                strings: &self.strings,
                slots: &slots,
                pointer_type,
                out_pointer,
                builder: &mut builder,
                function,
                return_type: return_type.clone(),
            };

            for (block_index, ir_block) in function.blocks.iter().enumerate() {
                if block_index != function.entry {
                    translator
                        .builder
                        .switch_to_block(clif_blocks[block_index]);
                }
                for statement in &ir_block.statements {
                    translator.statement(statement)?;
                }
                translator.terminator(&ir_block.terminator, &clif_blocks)?;
            }
        }

        builder.seal_all_blocks();
        builder.finalize();
        Ok(func_id)
    }
}

fn is_aggregate(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Struct(_)
            | Type::Enum(_)
            | Type::Array(_, _)
            | Type::Str
            | Type::Slice(_)
    )
}

// The machine type one returned register holds. A register carrying an odd
// number of bytes is widened to the next one that exists, and only the bytes
// the aggregate has are written back out.
fn register_type(register: CRegister) -> types::Type {
    if register.float {
        return if register.bytes <= 4 {
            types::F32
        } else {
            types::F64
        };
    }
    match register.bytes {
        1 => types::I8,
        2 => types::I16,
        3 | 4 => types::I32,
        _ => types::I64,
    }
}

// Write one incoming register into the struct this function is collecting. The
// mirror of store_returned_register: a whole number of bytes a store can express
// goes straight in, and anything else (a System V eightbyte with three bytes in
// it) goes through a scratch slot so that not one byte past the aggregate is
// written.
#[allow(clippy::too_many_arguments)]
fn store_incoming_register(
    builder: &mut FunctionBuilder,
    register: CRegister,
    value: Value,
    base: Value,
    layout: &CLayout,
    pointer_type: types::Type,
    memcpy: FuncId,
    decls: &Decls,
) {
    let bytes = register.bytes.min(layout.size - register.offset);
    if matches!(bytes, 1 | 2 | 4 | 8) && bytes == register.bytes {
        let at = builder.ins().iadd_imm(base, register.offset as i64);
        builder.ins().store(MemFlags::new(), value, at, 0);
        return;
    }
    let slot = builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        8,
        3,
    ));
    builder.ins().stack_store(value, slot, 0);
    let source = builder.ins().stack_addr(pointer_type, slot, 0);
    let destination = builder.ins().iadd_imm(base, register.offset as i64);
    let size = builder.ins().iconst(pointer_type, bytes as i64);
    let memcpy_ref = decls.declare_func_in_func(memcpy, builder.func);
    builder.ins().call(memcpy_ref, &[destination, source, size]);
}

// A call through a function pointer, as the pieces the emitter needs. They
// travel together because they describe one signature.
struct IndirectCall<'a> {
    callee: &'a IrOperand,
    arguments: &'a [IrOperand],
    parameter_types: &'a [Type],
    return_type: &'a Type,
}

fn param_abi_type(pointer_type: types::Type, ty: &Type) -> Result<types::Type> {
    if is_aggregate(ty) {
        Ok(pointer_type)
    } else {
        clif_type(pointer_type, ty)
    }
}

fn clif_type(pointer_type: types::Type, ty: &Type) -> Result<types::Type> {
    Ok(match ty {
        Type::I8 | Type::U8 | Type::Bool => types::I8,
        Type::I16 | Type::U16 => types::I16,
        Type::I32 | Type::U32 => types::I32,
        Type::I64 | Type::U64 | Type::Isize | Type::Usize => types::I64,
        Type::Handle(_) => types::I64,
        Type::F32 => types::F32,
        Type::F64 => types::F64,
        Type::Ptr(_) | Type::Ref(_) | Type::RefMut(_) | Type::Proc(_, _) => {
            pointer_type
        }
        Type::Distinct(_, inner) => clif_type(pointer_type, inner)?,
        other => {
            bail!("native backend: type not supported in codegen: {other}")
        }
    })
}

struct Translator<'a, 'b> {
    decls: &'a Decls,
    functions: &'a HashMap<String, FuncId>,
    strings: &'a HashMap<String, DataId>,
    slots: &'a HashMap<usize, StackSlot>,
    pointer_type: types::Type,
    out_pointer: Option<Value>,
    builder: &'a mut FunctionBuilder<'b>,
    function: &'a IrFunction,
    return_type: Type,
}

impl Translator<'_, '_> {
    fn slot_address(&mut self, local: usize) -> Result<Value> {
        let slot = self.slots.get(&local).ok_or_else(|| {
            anyhow::anyhow!(
                "_{local} in '{}' is a {} and has no storage to take the address of",
                self.function.name,
                self.function.local_type(local)
            )
        })?;
        Ok(self.builder.ins().stack_addr(self.pointer_type, *slot, 0))
    }

    // Where an aggregate a local names lives. A local of aggregate type is its
    // own storage. A local of reference type holds the address of someone
    // else's, which is what a borrowed parameter is, so its value is already
    // the address and taking its slot address would be taking the address of
    // the pointer.
    fn aggregate_address(&mut self, local: usize) -> Result<Value> {
        if matches!(
            self.function.local_type(local),
            Type::Ref(_) | Type::RefMut(_)
        ) {
            return self.operand(&IrOperand::Local(local));
        }
        self.slot_address(local)
    }

    fn emit_memcpy(&mut self, destination: Value, source: Value, size: usize) {
        let size_value =
            self.builder.ins().iconst(self.pointer_type, size as i64);
        let memcpy = self.functions["memcpy"];
        let memcpy_ref =
            self.decls.declare_func_in_func(memcpy, self.builder.func);
        self.builder
            .ins()
            .call(memcpy_ref, &[destination, source, size_value]);
    }

    fn statement(&mut self, statement: &IrStatement) -> Result<()> {
        match statement {
            IrStatement::Assign(local, rvalue) => {
                let local_type = self.function.local_type(*local).clone();
                if matches!(local_type, Type::Void | Type::Unknown) {
                    match rvalue {
                        IrRvalue::Call {
                            function,
                            arguments,
                        } => {
                            self.emit_call(function, arguments)?;
                        }
                        IrRvalue::CallIndirect {
                            callee,
                            arguments,
                            parameter_types,
                            return_type,
                        } => {
                            self.emit_call_indirect(
                                IndirectCall {
                                    callee,
                                    arguments,
                                    parameter_types,
                                    return_type,
                                },
                                None,
                            )?;
                        }
                        _ => {}
                    }
                    return Ok(());
                }
                if is_aggregate(&local_type) {
                    match rvalue {
                        IrRvalue::Use(IrOperand::Local(source)) => {
                            let destination = self.slot_address(*local)?;
                            let source_address =
                                self.aggregate_address(*source)?;
                            let size = self.function.locals[*local].size;
                            self.emit_memcpy(destination, source_address, size);
                        }
                        IrRvalue::Call {
                            function,
                            arguments,
                        } => {
                            let out = self.slot_address(*local)?;
                            self.emit_call_with_out(function, arguments, out)?;
                        }
                        // A function pointer hands an aggregate back the way a
                        // named function does, through the trailing
                        // out-pointer, since both are Frost functions and the
                        // pointer's type is the signature.
                        IrRvalue::CallIndirect {
                            callee,
                            arguments,
                            parameter_types,
                            return_type,
                        } => {
                            let out = self.slot_address(*local)?;
                            self.emit_call_indirect(
                                IndirectCall {
                                    callee,
                                    arguments,
                                    parameter_types,
                                    return_type,
                                },
                                Some(out),
                            )?;
                        }
                        _ => bail!(
                            "native backend: unsupported aggregate assignment"
                        ),
                    }
                    return Ok(());
                }
                let value = self.rvalue(rvalue, &local_type)?;
                if let Some(slot) = self.slots.get(local) {
                    self.builder.ins().stack_store(value, *slot, 0);
                } else {
                    self.builder.def_var(Variable::new(*local), value);
                }
                Ok(())
            }
            IrStatement::Store { address, value } => {
                let address_value = self.operand(address)?;
                let value_value = self.operand(value)?;
                self.builder.ins().store(
                    MemFlags::new(),
                    value_value,
                    address_value,
                    0,
                );
                Ok(())
            }
            IrStatement::Copy {
                destination,
                source,
                size,
            } => {
                let destination_value = self.operand(destination)?;
                let source_value = self.operand(source)?;
                self.emit_memcpy(destination_value, source_value, *size);
                Ok(())
            }
            IrStatement::Own(_) | IrStatement::Consume(_) => Ok(()),
        }
    }

    fn rvalue(
        &mut self,
        rvalue: &IrRvalue,
        result_type: &Type,
    ) -> Result<Value> {
        match rvalue {
            IrRvalue::Use(operand) => self.operand(operand),
            IrRvalue::Unary(op, operand) => {
                let value = self.operand(operand)?;
                let operand_type = self.operand_type(operand);
                Ok(match op {
                    IrUnOp::Negate => {
                        if operand_type.is_float() {
                            self.builder.ins().fneg(value)
                        } else {
                            self.builder.ins().ineg(value)
                        }
                    }
                    IrUnOp::Not => self.builder.ins().bxor_imm(value, 1),
                })
            }
            IrRvalue::Binary(op, left, right) => {
                let operand_type = self.operand_type(left);
                let left_value = self.operand(left)?;
                let right_value = self.operand(right)?;
                self.binary(*op, left_value, right_value, &operand_type)
            }
            IrRvalue::Cast(operand, target) => {
                let value = self.operand(operand)?;
                let source = self.operand_type(operand);
                self.cast(value, &source, target)
            }
            IrRvalue::AddressOf { local, offset } => {
                let Some(slot) = self.slots.get(local) else {
                    bail!(
                        "native backend: address taken of a non-memory local"
                    );
                };
                Ok(self.builder.ins().stack_addr(
                    self.pointer_type,
                    *slot,
                    *offset as i32,
                ))
            }
            IrRvalue::FieldAddress { base, offset } => {
                let base_value = self.operand(base)?;
                if *offset == 0 {
                    Ok(base_value)
                } else {
                    Ok(self.builder.ins().iadd_imm(base_value, *offset as i64))
                }
            }
            IrRvalue::ElementAddress {
                base,
                index,
                element_size,
            } => {
                let base_value = self.operand(base)?;
                let index_value = self.operand(index)?;
                let scaled = if *element_size == 1 {
                    index_value
                } else {
                    self.builder
                        .ins()
                        .imul_imm(index_value, *element_size as i64)
                };
                Ok(self.builder.ins().iadd(base_value, scaled))
            }
            IrRvalue::Load { address, ty } => {
                let address_value = self.operand(address)?;
                let clif = clif_type(self.pointer_type, ty)?;
                Ok(self.builder.ins().load(
                    clif,
                    MemFlags::new(),
                    address_value,
                    0,
                ))
            }
            IrRvalue::Call {
                function,
                arguments,
            } => {
                let results = self.emit_call(function, arguments)?;
                match results.first() {
                    Some(value) => Ok(*value),
                    None => Ok(self.zero_value(result_type)?),
                }
            }
            IrRvalue::FunctionAddress(name) => {
                let Some(func_id) = self.functions.get(name) else {
                    bail!(
                        "native backend: address of undeclared function '{name}'"
                    );
                };
                let func_ref = self
                    .decls
                    .declare_func_in_func(*func_id, self.builder.func);
                Ok(self.builder.ins().func_addr(self.pointer_type, func_ref))
            }
            IrRvalue::CallIndirect {
                callee,
                arguments,
                parameter_types,
                return_type,
            } => {
                let results = self.emit_call_indirect(
                    IndirectCall {
                        callee,
                        arguments,
                        parameter_types,
                        return_type,
                    },
                    None,
                )?;
                match results.first() {
                    Some(value) => Ok(*value),
                    None => Ok(self.zero_value(result_type)?),
                }
            }
        }
    }

    fn binary(
        &mut self,
        op: IrBinOp,
        left: Value,
        right: Value,
        operand_type: &Type,
    ) -> Result<Value> {
        let float = operand_type.is_float();
        let signed = is_signed(operand_type);
        let instructions = self.builder.ins();
        Ok(match op {
            IrBinOp::Add if float => instructions.fadd(left, right),
            IrBinOp::Add => instructions.iadd(left, right),
            IrBinOp::Subtract if float => instructions.fsub(left, right),
            IrBinOp::Subtract => instructions.isub(left, right),
            IrBinOp::Multiply if float => instructions.fmul(left, right),
            IrBinOp::Multiply => instructions.imul(left, right),
            IrBinOp::Divide if float => instructions.fdiv(left, right),
            IrBinOp::Divide if signed => instructions.sdiv(left, right),
            IrBinOp::Divide => instructions.udiv(left, right),
            IrBinOp::Modulo if signed => instructions.srem(left, right),
            IrBinOp::Modulo => instructions.urem(left, right),
            IrBinOp::BitwiseAnd => instructions.band(left, right),
            IrBinOp::BitwiseOr => instructions.bor(left, right),
            IrBinOp::ShiftLeft => instructions.ishl(left, right),
            IrBinOp::ShiftRight if signed => instructions.sshr(left, right),
            IrBinOp::ShiftRight => instructions.ushr(left, right),
            comparison => {
                return self.comparison(comparison, left, right, operand_type);
            }
        })
    }

    fn comparison(
        &mut self,
        op: IrBinOp,
        left: Value,
        right: Value,
        operand_type: &Type,
    ) -> Result<Value> {
        let float = operand_type.is_float();
        let signed = is_signed(operand_type);
        if float {
            let condition = match op {
                IrBinOp::Equal => FloatCC::Equal,
                IrBinOp::NotEqual => FloatCC::NotEqual,
                IrBinOp::LessThan => FloatCC::LessThan,
                IrBinOp::LessThanOrEqual => FloatCC::LessThanOrEqual,
                IrBinOp::GreaterThan => FloatCC::GreaterThan,
                IrBinOp::GreaterThanOrEqual => FloatCC::GreaterThanOrEqual,
                _ => bail!("native backend: invalid float comparison"),
            };
            return Ok(self.builder.ins().fcmp(condition, left, right));
        }
        let condition = match (op, signed) {
            (IrBinOp::Equal, _) => IntCC::Equal,
            (IrBinOp::NotEqual, _) => IntCC::NotEqual,
            (IrBinOp::LessThan, true) => IntCC::SignedLessThan,
            (IrBinOp::LessThan, false) => IntCC::UnsignedLessThan,
            (IrBinOp::LessThanOrEqual, true) => IntCC::SignedLessThanOrEqual,
            (IrBinOp::LessThanOrEqual, false) => IntCC::UnsignedLessThanOrEqual,
            (IrBinOp::GreaterThan, true) => IntCC::SignedGreaterThan,
            (IrBinOp::GreaterThan, false) => IntCC::UnsignedGreaterThan,
            (IrBinOp::GreaterThanOrEqual, true) => {
                IntCC::SignedGreaterThanOrEqual
            }
            (IrBinOp::GreaterThanOrEqual, false) => {
                IntCC::UnsignedGreaterThanOrEqual
            }
            _ => bail!("native backend: invalid integer comparison"),
        };
        Ok(self.builder.ins().icmp(condition, left, right))
    }

    fn cast(
        &mut self,
        value: Value,
        source: &Type,
        target: &Type,
    ) -> Result<Value> {
        let source_clif = clif_type(self.pointer_type, source)?;
        let target_clif = clif_type(self.pointer_type, target)?;
        if source_clif == target_clif {
            return Ok(value);
        }
        let source_float = source.is_float();
        let target_float = target.is_float();
        Ok(match (source_float, target_float) {
            (false, false) => {
                if target_clif.bits() > source_clif.bits() {
                    if is_signed(source) {
                        self.builder.ins().sextend(target_clif, value)
                    } else {
                        self.builder.ins().uextend(target_clif, value)
                    }
                } else {
                    self.builder.ins().ireduce(target_clif, value)
                }
            }
            (false, true) => {
                if is_signed(source) {
                    self.builder.ins().fcvt_from_sint(target_clif, value)
                } else {
                    self.builder.ins().fcvt_from_uint(target_clif, value)
                }
            }
            (true, false) => {
                if is_signed(target) {
                    self.builder.ins().fcvt_to_sint(target_clif, value)
                } else {
                    self.builder.ins().fcvt_to_uint(target_clif, value)
                }
            }
            (true, true) => {
                if target_clif.bits() > source_clif.bits() {
                    self.builder.ins().fpromote(target_clif, value)
                } else {
                    self.builder.ins().fdemote(target_clif, value)
                }
            }
        })
    }

    fn emit_call(
        &mut self,
        function: &str,
        arguments: &[IrOperand],
    ) -> Result<Vec<Value>> {
        if function == BOUNDS_CHECK && arguments.len() == 2 {
            self.emit_bounds_check(arguments)?;
            return Ok(Vec::new());
        }
        let Some(func_id) = self.functions.get(function) else {
            bail!("native backend: call to undeclared function '{function}'");
        };
        let func_ref =
            self.decls.declare_func_in_func(*func_id, self.builder.func);
        let argument_values = self.call_arguments(function, arguments)?;
        let call = self.builder.ins().call(func_ref, &argument_values);
        Ok(self.builder.inst_results(call).to_vec())
    }

    // An index check as a compare and a branch. The runtime holds the message
    // and the abort, and an index in range reaches neither, so what a checked
    // read costs is the compare rather than a call with its arguments. The
    // comparison is unsigned, which is what makes one of them answer for a
    // negative index as well as one past the end.
    fn emit_bounds_check(&mut self, arguments: &[IrOperand]) -> Result<()> {
        let Some(func_id) = self.functions.get(BOUNDS_CHECK) else {
            bail!(
                "native backend: call to undeclared function '{BOUNDS_CHECK}'"
            );
        };
        let func_ref =
            self.decls.declare_func_in_func(*func_id, self.builder.func);
        let index = self.operand(&arguments[0])?;
        let length = self.operand(&arguments[1])?;
        let outside = self.builder.ins().icmp(
            IntCC::UnsignedGreaterThanOrEqual,
            index,
            length,
        );
        let report = self.builder.create_block();
        let past = self.builder.create_block();
        self.builder.ins().brif(outside, report, &[], past, &[]);
        self.builder.switch_to_block(report);
        self.builder.ins().call(func_ref, &[index, length]);
        self.builder.ins().jump(past, &[]);
        self.builder.switch_to_block(past);
        Ok(())
    }

    // The values a call passes. Every aggregate is a pointer in the IR, so a
    // parameter the callee takes by value is the one place that pointer is
    // taken apart: into the registers the target splits the struct across, or
    // into a copy the caller owns and passes the address of.
    fn call_arguments(
        &mut self,
        function: &str,
        arguments: &[IrOperand],
    ) -> Result<Vec<Value>> {
        let by_value = self.decls.c_arguments.get(function).cloned();
        let Some(by_value) = by_value else {
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments {
                values.push(self.operand(argument)?);
            }
            return Ok(values);
        };
        let mut values = Vec::with_capacity(arguments.len());
        for (index, argument) in arguments.iter().enumerate() {
            let address = self.operand(argument)?;
            let Some((_, layout, passed)) =
                by_value.iter().find(|(at, _, _)| *at == index)
            else {
                values.push(address);
                continue;
            };
            match passed {
                CArgument::Registers(registers) => {
                    for register in registers {
                        values.push(self.load_argument_register(
                            *register, address, layout,
                        ));
                    }
                }
                // The callee's parameter is its own, so what it is handed is
                // the address of a copy. Handing over the caller's value would
                // let a callee that writes to its parameter write through to
                // the caller's, which passing by value does not do.
                CArgument::Indirect => {
                    let slot = self.builder.create_sized_stack_slot(
                        StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            layout.size.max(1) as u32,
                            4,
                        ),
                    );
                    let copy = self.builder.ins().stack_addr(
                        self.pointer_type,
                        slot,
                        0,
                    );
                    self.emit_memcpy(copy, address, layout.size);
                    values.push(copy);
                }
                // Cranelift copies the bytes into the argument area itself,
                // given their address, so this is the one shape that needs no
                // copy of its own.
                CArgument::Stack => values.push(address),
            }
        }
        Ok(values)
    }

    // Read one register's worth of an aggregate out of the caller's storage.
    // The mirror of store_returned_register: a whole number of bytes a load can
    // express is loaded directly, and anything else goes through a scratch slot
    // so that not one byte past the aggregate is read.
    fn load_argument_register(
        &mut self,
        register: CRegister,
        address: Value,
        layout: &CLayout,
    ) -> Value {
        let bytes = register.bytes.min(layout.size - register.offset);
        let clif = register_type(register);
        if matches!(bytes, 1 | 2 | 4 | 8) && bytes == register.bytes {
            let at =
                self.builder.ins().iadd_imm(address, register.offset as i64);
            return self.builder.ins().load(clif, MemFlags::new(), at, 0);
        }
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8,
            3,
        ));
        let scratch = self.builder.ins().stack_addr(self.pointer_type, slot, 0);
        let zero = self.builder.ins().iconst(types::I64, 0);
        self.builder.ins().stack_store(zero, slot, 0);
        let source =
            self.builder.ins().iadd_imm(address, register.offset as i64);
        self.emit_memcpy(scratch, source, bytes);
        self.builder.ins().stack_load(clif, slot, 0)
    }

    // `out` is where an aggregate result lands. A Frost function that returns
    // one takes the address as a trailing parameter and returns nothing, and a
    // call through a pointer is the same call, so the signature built here
    // matches the one build_signature builds for the callee.
    fn emit_call_indirect(
        &mut self,
        call: IndirectCall,
        out: Option<Value>,
    ) -> Result<Vec<Value>> {
        let IndirectCall {
            callee,
            arguments,
            parameter_types,
            return_type,
        } = call;
        let mut signature = self.decls.make_signature();
        for parameter in parameter_types {
            signature.params.push(AbiParam::new(param_abi_type(
                self.pointer_type,
                parameter,
            )?));
        }
        if out.is_some() {
            signature.params.push(AbiParam::new(self.pointer_type));
        } else if !matches!(return_type, Type::Void) {
            signature.returns.push(AbiParam::new(clif_type(
                self.pointer_type,
                return_type,
            )?));
        }
        let signature_ref = self.builder.import_signature(signature);
        let callee_value = self.operand(callee)?;
        let mut argument_values = Vec::with_capacity(arguments.len() + 1);
        for argument in arguments {
            argument_values.push(self.operand(argument)?);
        }
        if let Some(out) = out {
            argument_values.push(out);
        }
        let call = self.builder.ins().call_indirect(
            signature_ref,
            callee_value,
            &argument_values,
        );
        Ok(self.builder.inst_results(call).to_vec())
    }

    fn emit_call_with_out(
        &mut self,
        function: &str,
        arguments: &[IrOperand],
        out: Value,
    ) -> Result<()> {
        let Some(func_id) = self.functions.get(function) else {
            bail!("native backend: call to undeclared function '{function}'");
        };
        let func_ref =
            self.decls.declare_func_in_func(*func_id, self.builder.func);
        // A C function returning an aggregate does not take Frost's trailing
        // out-pointer. Either it takes a hidden pointer first, or it hands the
        // value back in registers and the caller writes it into its own
        // storage. See src/c_abi.rs.
        if let Some((layout, returned)) = self.decls.c_returns.get(function) {
            let layout = layout.clone();
            let returned = returned.clone();
            let mut argument_values = Vec::with_capacity(arguments.len() + 1);
            if matches!(returned, CReturn::Indirect) {
                argument_values.push(out);
            }
            argument_values.extend(self.call_arguments(function, arguments)?);
            let call = self.builder.ins().call(func_ref, &argument_values);
            if let CReturn::Registers(registers) = returned {
                let results = self.builder.inst_results(call).to_vec();
                for (register, value) in registers.iter().zip(results) {
                    self.store_returned_register(
                        *register, value, out, &layout,
                    );
                }
            }
            return Ok(());
        }
        let mut argument_values = Vec::with_capacity(arguments.len() + 1);
        for argument in arguments {
            argument_values.push(self.operand(argument)?);
        }
        argument_values.push(out);
        self.builder.ins().call(func_ref, &argument_values);
        Ok(())
    }

    // Write one returned register into the caller's storage. A register holding
    // a whole number of bytes that a store can express is stored directly.
    // Anything else (a System V eightbyte with three bytes in it, say) goes
    // through a scratch slot so that not one byte past the aggregate is
    // written.
    fn store_returned_register(
        &mut self,
        register: CRegister,
        value: Value,
        out: Value,
        layout: &CLayout,
    ) {
        let bytes = register.bytes.min(layout.size - register.offset);
        if matches!(bytes, 1 | 2 | 4 | 8) {
            let address =
                self.builder.ins().iadd_imm(out, register.offset as i64);
            self.builder.ins().store(MemFlags::new(), value, address, 0);
            return;
        }
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            8,
            3,
        ));
        self.builder.ins().stack_store(value, slot, 0);
        let source = self.builder.ins().stack_addr(self.pointer_type, slot, 0);
        let destination =
            self.builder.ins().iadd_imm(out, register.offset as i64);
        self.emit_memcpy(destination, source, bytes);
    }

    fn terminator(
        &mut self,
        terminator: &IrTerminator,
        blocks: &[Block],
    ) -> Result<()> {
        match terminator {
            IrTerminator::Return(None) => {
                self.emit_return(None)?;
            }
            IrTerminator::Return(Some(operand)) => {
                self.emit_return(Some(operand))?;
            }
            IrTerminator::Jump(block) => {
                self.builder.ins().jump(blocks[*block], &[]);
            }
            IrTerminator::Branch {
                condition,
                then_block,
                else_block,
            } => {
                let condition_value = self.operand(condition)?;
                self.builder.ins().brif(
                    condition_value,
                    blocks[*then_block],
                    &[],
                    blocks[*else_block],
                    &[],
                );
            }
            IrTerminator::Unreachable => {
                self.emit_return(None)?;
            }
        }
        Ok(())
    }

    fn emit_return(&mut self, operand: Option<&IrOperand>) -> Result<()> {
        if let Some(out_pointer) = self.out_pointer {
            if let Some(IrOperand::Local(source)) = operand {
                // Through the same door as every other aggregate read: a
                // borrowed parameter already holds the address, and asking for
                // its slot asked for the address of the pointer. Returning one
                // straight back, `fn(held: Small) -> Small { held }`, was
                // refused for want of storage that was never needed.
                let source_address = self.aggregate_address(*source)?;
                let size = self.function.locals[*source].size;
                self.emit_memcpy(out_pointer, source_address, size);
            }
            self.builder.ins().return_(&[]);
            return Ok(());
        }
        if matches!(self.return_type, Type::Void) {
            self.builder.ins().return_(&[]);
            return Ok(());
        }
        let value = match operand {
            Some(operand) => {
                let source = self.operand_type(operand);
                let value = self.operand(operand)?;
                self.cast(value, &source, &self.return_type.clone())?
            }
            None => self.zero_value(&self.return_type.clone())?,
        };
        self.builder.ins().return_(&[value]);
        Ok(())
    }

    fn operand(&mut self, operand: &IrOperand) -> Result<Value> {
        match operand {
            IrOperand::Local(local) => {
                if let Some(slot) = self.slots.get(local) {
                    let clif = clif_type(
                        self.pointer_type,
                        self.function.local_type(*local),
                    )?;
                    Ok(self.builder.ins().stack_load(clif, *slot, 0))
                } else {
                    Ok(self.builder.use_var(Variable::new(*local)))
                }
            }
            IrOperand::Constant(constant) => self.constant(constant),
        }
    }

    fn constant(&mut self, constant: &IrConstant) -> Result<Value> {
        match constant {
            IrConstant::Integer(value, ty) => {
                let clif = clif_type(self.pointer_type, ty)?;
                Ok(self.builder.ins().iconst(clif, *value))
            }
            IrConstant::Float(value, Type::F32) => {
                Ok(self.builder.ins().f32const(*value as f32))
            }
            IrConstant::Float(value, _) => {
                Ok(self.builder.ins().f64const(*value))
            }
            IrConstant::Bool(value) => {
                Ok(self.builder.ins().iconst(types::I8, i64::from(*value)))
            }
            IrConstant::CString(text) => {
                let data_id = self.strings[text];
                let local =
                    self.decls.declare_data_in_func(data_id, self.builder.func);
                Ok(self.builder.ins().symbol_value(self.pointer_type, local))
            }
            IrConstant::Unit => {
                bail!(
                    "native backend: unit value used as a real value in '{}'",
                    self.function.name
                )
            }
        }
    }

    fn zero_value(&mut self, ty: &Type) -> Result<Value> {
        let clif = clif_type(self.pointer_type, ty)?;
        Ok(match ty {
            Type::F32 => self.builder.ins().f32const(0.0),
            Type::F64 => self.builder.ins().f64const(0.0),
            _ => self.builder.ins().iconst(clif, 0),
        })
    }

    fn operand_type(&self, operand: &IrOperand) -> Type {
        match operand {
            IrOperand::Local(local) => self.function.local_type(*local).clone(),
            IrOperand::Constant(constant) => constant.constant_type(),
        }
    }
}

// A distinct type computes as what it is represented by, so this looks through
// it, the same way `Type::is_float` does. Missing that emitted an integer
// subtract for two `distinct f64` values, which Cranelift's verifier caught and
// nothing else would have.
fn is_signed(ty: &Type) -> bool {
    match ty {
        Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Isize => true,
        Type::Distinct(_, inner) => is_signed(inner),
        _ => false,
    }
}

fn collect_strings(
    module: &IrModule,
    handle: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    for function in &module.functions {
        for block in &function.blocks {
            for statement in &block.statements {
                match statement {
                    IrStatement::Assign(_, rvalue) => {
                        collect_rvalue_strings(rvalue, handle)?;
                    }
                    IrStatement::Store { address, value } => {
                        collect_operand_strings(address, handle)?;
                        collect_operand_strings(value, handle)?;
                    }
                    IrStatement::Copy {
                        destination,
                        source,
                        ..
                    } => {
                        collect_operand_strings(destination, handle)?;
                        collect_operand_strings(source, handle)?;
                    }
                    IrStatement::Own(_) | IrStatement::Consume(_) => {}
                }
            }
            if let IrTerminator::Return(Some(operand)) = &block.terminator {
                collect_operand_strings(operand, handle)?;
            }
        }
    }
    Ok(())
}

fn collect_rvalue_strings(
    rvalue: &IrRvalue,
    handle: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    match rvalue {
        IrRvalue::Use(operand) | IrRvalue::Unary(_, operand) => {
            collect_operand_strings(operand, handle)
        }
        IrRvalue::Cast(operand, _) => collect_operand_strings(operand, handle),
        IrRvalue::Load { address, .. } => {
            collect_operand_strings(address, handle)
        }
        IrRvalue::FieldAddress { base, .. } => {
            collect_operand_strings(base, handle)
        }
        IrRvalue::ElementAddress { base, index, .. } => {
            collect_operand_strings(base, handle)?;
            collect_operand_strings(index, handle)
        }
        IrRvalue::AddressOf { .. } => Ok(()),
        IrRvalue::Binary(_, left, right) => {
            collect_operand_strings(left, handle)?;
            collect_operand_strings(right, handle)
        }
        IrRvalue::Call { arguments, .. }
        | IrRvalue::CallIndirect { arguments, .. } => {
            for argument in arguments {
                collect_operand_strings(argument, handle)?;
            }
            Ok(())
        }
        IrRvalue::FunctionAddress(_) => Ok(()),
    }
}

fn collect_operand_strings(
    operand: &IrOperand,
    handle: &mut impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    if let IrOperand::Constant(IrConstant::CString(text)) = operand {
        handle(text)?;
    }
    Ok(())
}
