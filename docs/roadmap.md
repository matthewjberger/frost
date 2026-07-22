# What is left, and the order to do it in

Goal 8 in [philosophy.md](philosophy.md) makes compilation speed a promise rather
than a happy accident, and that promise has a bill. This is the bill, sequenced
so that nothing here gets built twice.

## The target

Competitive with Jai and Odin, which in practice means a full build in the
100,000 lines per second range rather than merely "fast for a compiler". That is
a number to measure against, not a feeling.

Where Frost stands today, from `just bench-scaling` on 58,107 lines:

| stage | rate |
| --- | --- |
| front end (`--emit-c`, 318 ms) | ~183,000 lines/sec |
| full build (`--native`, 349 ms) | ~166,000 lines/sec |

Both clear the bar, and the backend is no longer the gap: on that program code
generation is 64 ms of a 349 ms build, with the front end holding the rest. That
tells you what not to work on next. Cranelift is done being the problem.

Before item 1 landed the same program took 1.11 s, or about 52,000 lines per
second, and the backend was 1,285 ms of it. The remaining work on speed is the
shape question, item 3, not a constant factor.

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

## 2. Bounds on compile-time arguments (done)

A compile-time function parameter may now say what signature it needs:

```
best :: fn($T: Type, $before: fn(T, T) -> bool, move x: $T, move y: $T) -> $T
```

`$T: Type` still means a type. `$f: fn(...) -> ...` means a function, and the
argument's declared signature is checked against it, with the call's own
substitution applied, so `T` in the bound means what it means at the call.
Passing a type where a function is declared, or a function whose signature does
not match, is reported against the parameter list rather than against a line
inside the specialized body the reader never wrote.

**What it is not.** Not a trait system. No coherence, no orphan rules, no solver.
A comparison of one signature against another, on one parameter kind. See the
"Not trait-based" non-goal.

**The catch, since it will bite again.** `is_type_parameter` keys on the
annotation *being* `Type::TypeParam(name)`, so putting the signature there flips
the parameter to a runtime parameter and breaks every generic. It lives in a
separate `compile_time_signature: Option<Type>` on `Parameter`, beside the
annotation rather than replacing it.

The check is deferred until every argument has been walked, because a bound can
name type parameters that later value arguments are what bind.

**Still open**, and deliberately: `$T: Type` has no bound of its own, so
`double :: fn(v: $T) -> T` still requires `T` to be numeric silently. Bounding a
type rather than a function is the thing that turns into a trait system if it is
approached carelessly, and nothing needs it yet.

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

## 1. Parallel code generation (done)

Code generation runs on every thread the machine has, correctness is carried by
the differential oracle, and both self-hosting fixpoints still hold byte for
byte. The sweep, with `FROST_THREADS=n`, on 10,401 functions:

| threads | 1 | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- | --- |
| code generation | 385 ms | 218 ms | 111 ms | 65 ms | 55 ms |

That is 7.0x on a machine with eight physical cores, which is about as close to
linear as this gets.

**It took two false starts, both worth recording.** The first landed version won
only 1.2x from sixteen threads, and a processor-time-against-wall-time reading
said total CPU was roughly flat across thread counts. That reading was correct
and the conclusion drawn from it, that the threads were somehow serialized, was
wrong. What settled it was a timer inside each worker:

```
worker 641 functions, build 2 ms, compile 25 ms, wall 28 ms      (x15)
worker 641 functions, build 30 ms, compile 989 ms, wall 1019 ms   (x1)
```

Fifteen threads did their share in 28 ms. One took 989 ms for the same count of
functions. Nothing was serialized; one function was 97% of the work, because
`bench/generate.py` emitted a `main` with ten thousand call sites and a single
function cannot be split across threads. Fifteen threads doing 28 ms each adds
only ~400 ms of CPU over a 1,000 ms wall, which is exactly the 1.3x ratio that
had been read as evidence of serialization. The benchmark was the bug.

The generator now fans its calls through intermediate functions, which is the
shape real code has. That alone took 1,021 ms to 211 ms.

**The second false start** was static chunking. With `main` fixed the expensive
functions still clustered, since `chunks(per_thread)` is contiguous and the
module's function list is ordered, so one thread got all of them: 209 ms against
31 ms for the rest. Cost per function varies by more than an order of magnitude,
so the split is now a shared atomic cursor handing out one function at a time,
with results sorted back into module order so the object does not depend on how
the threads interleaved. That took 211 ms to 55 ms.

The lesson worth keeping is that both wrong turns came from trusting a summary
statistic over a per-worker measurement. Whole-process CPU time could not
distinguish "threads are serialized" from "one thread has all the work", and
those want opposite fixes.

What else helped and is kept: not cloning each function's IR out of the context
to hand to `define_function_bytes`. That backend uses the function only to
resolve relocation targets against its imported names, so the value is moved out
with `mem::replace` rather than deep copied, worth about 130 ms at the old size.

Ruled out by measurement along the way, so nobody re-runs them: serial
declaration (9 ms), serial defining (3 ms), object emission (3 ms), allocator
contention (mimalloc changed nothing), and one ISA per thread against a shared
one (flat, and kept anyway since it is free).

**The remaining serial floor** is a single large function, which no amount of
threading divides. It does not bind on real code, where the self-hosted compiler
is 244 functions and 6 ms of code generation, but it is the thing that would
bind first if it ever bound.

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
