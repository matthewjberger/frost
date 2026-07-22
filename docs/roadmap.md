# What is left, and the order to do it in

Goal 8 in [philosophy.md](philosophy.md) makes compilation speed a promise rather
than a happy accident, and that promise has a bill. This is the bill, sequenced
so that nothing here gets built twice.

## The target

Competitive with Jai and Odin, which in practice means a full build in the
100,000 lines per second range rather than merely "fast for a compiler". That is
a number to measure against, not a feeling.

Where Frost stands today, from `just bench-scaling` on 57,607 lines:

| stage | rate |
| --- | --- |
| front end (`--emit-c`, 395 ms) | ~146,000 lines/sec |
| full build (`--native`, 1.11 s) | ~52,000 lines/sec |

So the front end already clears the bar and the whole build is roughly half of
it, and the entire gap is the backend. That is worth stating plainly because it
tells you what not to work on: parse, parameter modes, regions, ownership, IR
lowering, type checking and monomorphization are not the problem and optimizing
them would be motion without progress. Cranelift and object emission are the
problem.

Measured further, with `FROST_TIMINGS=1`, the backend splits like this:

| program | code generation | object emission |
| --- | --- | --- |
| 6,401 functions | 779 ms | 1 ms |
| 10,241 functions | 1,285 ms | 4 ms |

Object emission is free. Essentially all of the backend is Cranelift compiling
one function at a time, which is per-function work on independent inputs. So
parallel code generation attacks the dominant cost directly, and it is the right
unit rather than the wrong one: even once modules become compilation units, the
functions inside a module still compile one at a time and still parallelize.

That measurement changed the order below. The first version of this document put
separate compilation first on the argument that per-function parallelism would be
designed twice. That argument was wrong, and it was wrong in the direction of
doing the harder thing first for a reason that did not survive contact with a
twenty-line measurement.

## 3. Separate compilation

**Why third rather than first.** It is the shape the speed goal requires *as
programs grow*, which is a different claim from the constant factor item 1
closes. Parallelism makes today's builds fast; this is what stops the curve
turning over. `src/imports.rs` flattens every import into one
AST, so a program's cost is whole-program by construction: monomorphization runs
to fixpoint over everything, and there is nothing to compile independently. Every
other scaling lever is a constant factor on top of that shape.

It also bounds monomorphization, which is the thing that actually grows. A
specialization need only be emitted once per module that needs it, rather than
once per program.

**The design decision to make.** What a module boundary means. The candidate that
fits the rest of the language: a module is a file, its interface is its `export`
line, and a specialization is emitted in the module that instantiates it. That
keeps the existing visibility rule as the interface rather than inventing a
second one.

**What it forces.** A generic's body has to be checkable without seeing every
caller, which is the one argument for bounds that Frost does not already have an
answer to. See item 2, which is why it comes next rather than first.

**Do not** start this by writing code. Start by writing down what a module's
compiled artifact contains and what has to be in it for a caller to compile
against it without the body.

## 2. Bounds on compile-time arguments

**Why second.** Item 1 needs it, and it is what the philosophy positively asks
for on its own. Goal 7 is predictability, and `double :: fn(v: $T) -> T` silently
requiring `T` to be numeric fails that at the signature, before any code runs.

**What it is.** `$compare: fn(T, T) -> bool` parses today and the annotation is
thrown away (`src/parser.rs`, generic-parameter position). Keep it, and check the
bound `ConstFn`'s signature against it at the call.

**What it is not.** Not a trait system. No coherence, no orphan rules, no solver.
A comparison of one signature against another, on one parameter kind. See the
"Not trait-based" non-goal.

**The catch.** `is_type_parameter` keys on the annotation *being*
`Type::TypeParam(name)`, so simply not discarding the annotation flips the
parameter to a runtime parameter and breaks every generic. It needs a separate
field on `Parameter`, something like `compile_time_signature: Option<Type>`, kept
beside the annotation rather than replacing it.

## 3. Callbacks with a typed context

**Why third.** It is the one place the implementation contradicts a goal rather
than merely falling short of an aspiration, so it is correctness rather than
performance. It is third only because items 1 and 2 change what a function
signature can say, and this design leans on that.

**The contradiction.** Goal 2 says safety comes from making dangerous shapes
unrepresentable. The only way to write a callback today is the C idiom, a
function pointer plus an untyped `^u8` userdata, which is long-lived, first
class, and outside every check in `src/regions.rs` and `src/ownership.rs`. So
every callback-shaped API in Frost is an unsafe API, not because callbacks are
unsafe but because the only expression of one is a raw escape hatch. That is the
inversion the surface `&` removal was meant to prevent, reappearing at the C
boundary.

**The design.** Closures stay a non-goal. Capture is not the answer; a written
down context is. A callback is a compile-time function argument plus a typed
context the caller owns:

```
Ctx :: struct { hits: i64 }

on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }

register :: extern fn($handler: Type, mut ctx: Ctx, ...) uses CallbackAbi
```

The compiler emits a trampoline for each `(handler, Ctx)` pair with the C ABI the
library expects, and that trampoline is the only code that casts the `^u8` back
to `^Ctx`. The cast still happens, because C requires it, but it is generated
once per instantiation inside code the user never writes, rather than appearing
at every call site in user code. The user writes a typed `mut ctx: Ctx` and the
perimeter holds everywhere a person is looking.

**What to settle before building it.** Who owns the context for the lifetime of
the registration, since it outlives the call that registers it. That is a region
question and `src/regions.rs` already has the machinery to ask it. The likely
answer is that registering borrows for longer than a call, which is the first
thing in the language that does, and it may want to be a linear obligation
(register/unregister as a consumed pair) rather than a borrow.

## 1. Parallel code generation (landed, and not yet paying)

**Landed, and it does not pay yet.** Code generation runs on every thread the
machine has, correctness is carried by the differential oracle, and the win is
1,285 ms to about 1,060 ms on sixteen threads at 10,241 functions. That is 1.2x
from 16x the threads, and it is not understood.

What has been ruled out by measurement, so nobody re-runs it:

| suspect | measured | verdict |
| --- | --- | --- |
| serial declaration | 9 ms | not it |
| serial defining | 3 ms | not it |
| object emission | 4 ms | not it |
| allocator contention | mimalloc changed nothing | not it |

So the serial tail is 16 ms against a second of parallel work, and Amdahl does
not explain it. The remaining candidates are that this machine's sixteen threads
are eight cores with SMT and Cranelift's compile is memory-bound enough that SMT
buys nothing, or that something inside `Context::compile` shares state. The next
step is a thread-count sweep, 1, 2, 4, 8, 16, which separates those two answers
in one run: memory-bound work flattens gradually, a shared lock flattens at two.

**Why it was first.** It is where the time is. Code generation is 1,285 ms of a 1,289 ms
backend at 10,241 functions, and the work is per-function on independent inputs,
so this is the one change that moves the number that misses the target. It is not
made redundant by separate compilation either, since functions within a module
still compile one at a time.

**The API path, already verified.** `Module::declare_func_in_func` only reads
`self.declarations()` despite taking `&mut self`, so it can be replicated against
an immutable snapshot. `define_function_bytes(func_id, func, alignment, bytes,
relocs)` exists in cranelift-module 0.116 and is the seam for handing back
separately compiled code. So: declare serially, build and `Context::compile` in
parallel, `define_function_bytes` serially.

**The shape, worked out.** `Generator` touches the module in fifteen places, but
only eight of them need the real thing, and they are all in `declare_strings`,
`declare_functions`, and the last two lines of `define_function`. The other seven
are `make_signature` and the two `*_in_func` helpers, which need nothing but the
call convention and the declarations. So the split is clean:

```rust
struct Decls {
    call_conv: isa::CallConv,
    // Signature and whether the symbol is colocated, per declared function.
    functions: HashMap<FuncId, (Signature, bool)>,
    data: HashMap<DataId, bool>,
}
```

with the two helpers replicating the module's own implementations exactly, both
verified against cranelift-module 0.116:

```rust
fn declare_func_in_func(&self, id: FuncId, func: &mut ir::Function) -> ir::FuncRef {
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
// data is the same with namespace 1, create_global_value and
// GlobalValueData::Symbol { offset: Imm64::new(0), tls: false }
```

Name the field `module` and the seven translation sites do not change at all.

**The three phases.** Declare serially as now. Then, per thread over a chunk of
functions, build the `ir::Function` and call
`context.compile(isa, &mut ControlPlane::default())`, keeping
`code.buffer.data().to_vec()`, `code.buffer.relocs().to_vec()` and
`code.alignment` so nothing borrows the context across the join. Then serially
`module.define_function_bytes(func_id, &func, alignment as u64, &bytes,
&relocs)`. `std::thread::scope` over chunks avoids taking a dependency on rayon
for what is one `map`.

Note that `Module::define_function` compiles inside itself, which is why the
compile has to move to `Context::compile` explicitly. That is the whole reason
this is a refactor rather than a loop change.

**The cost.** That is a real refactor of the backend's core, and the class
of bug it invites is the one that passes the test suite and corrupts output, the
way the sret register and the byte-stride bugs both did. Do it with the
differential oracle running, not after.

## Smaller things found and not yet fixed

- **Generic templates are re-parsed per instantiation in the self-hosted
  compiler**, now three times over rather than twice since concrete return types
  are computed for both backends. Parse the template to AST once and substitute.
  Listed as second-order lever 2 in [self-hosting.md](self-hosting.md).
- **An IR type error points into the generic's body rather than at the call
  site.** It carries a line and column now, but the line it names is inside code
  the reader did not write. Item 2 fixes the common case by catching it at the
  call instead.
- **The self-hosted compiler has no incremental or separate compilation
  either**, and item 1's design should say whether it is expected to grow one or
  whether the reference compiler is the only one that does.
