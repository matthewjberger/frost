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
second, and the backend was 1,285 ms of it.

The shape question, item 3, is closed too: a module is a compilation unit and
`--incremental` skips the ones an edit cannot reach. What is left on speed is
named at the end of that item, and it is a bounded piece of work rather than an
open question.

## 3. Separate compilation (done)

All five steps in [separate-compilation.md](separate-compilation.md) are built.
A module is a file, its interface is its `export` line, a specialization is
emitted in the module that instantiates it, each module is its own object on the
link path, and `--incremental` rebuilds a module only when its own source or an
imported interface changes.

From `just bench-incremental` on 9,484 lines across 65 files, one of which
changed: 668 ms full against 399 ms incremental, of which about 90 ms either way
is process start and the linker.

Three findings worth keeping, none of which were obvious going in:

- **A generic's body is part of its interface**, unavoidably, because the caller
  chooses the type arguments and so the caller instantiates the template. So
  changing a generic's body is an interface change and rebuilds every module
  that instantiates it, while changing an ordinary body is not. That distinction
  is what the fingerprint encodes and it is the whole reason the cache pays.
- **Private symbol names used to depend on import traversal order**, so a
  private `helper` was `__m3_helper` in one program and `__m7_helper` in
  another. A module's symbols have to be a property of the module before any of
  this is possible, which is why that was step 1 and why it was verifiable
  entirely on its own.
- **Cranelift has no weak or COMDAT linkage**, so duplicate specializations
  across modules are not folded and each module emits its own private copy.
  `FROST_MODULE_REPORT=1` measures how much that costs, and that measurement is
  the only thing that would justify revisiting it.

**What is still whole-program**, and it is the next lever on this axis: a
skipped module still contributes its interface, and an interface carries the
bodies of exported functions whether or not they are generic. So the front end
walks bodies it will not emit. Removing that needs a declaration form carrying a
full Frost signature and no body, which `extern` cannot be, and that is a change
to every pass that matches on a function declaration.

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

**Who owns the context, settled.** This was the open question, and the answer is
that **registration moves the context in and unregistration moves it back out**,
with the registration itself a `linear` value. Not a borrow.

```
Ctx :: struct { hits: i64 }
Registration :: linear struct { token: i64, ctx: Ctx }

on_event :: fn(mut ctx: Ctx, code: i64) { ctx.hits = ctx.hits + code }

register   :: extern fn($handler: Type, move ctx: Ctx) -> Registration uses CallbackAbi
unregister :: fn(move r: Registration) -> Ctx

run :: fn() -> i64 {
    r := register($on_event, Ctx { hits = 0 })
    pump_events()
    done := unregister(r)      // forgetting this is a compile error
    done.hits
}
```

Three things fall out, and each is a reason to prefer this over a borrow.

- **No new machinery.** A borrow that outlives its call would be the first thing
  in the language that does, and inventing it means inventing the region
  annotation the whole design is built on not having. Moving needs nothing new:
  `check_ownership` already stops the caller touching a moved value, and
  `check_linearity` already forces a `linear` value to be consumed exactly once.
- **The aliasing guarantee is the one you want.** While registered, the callback
  may fire at any moment, so the caller must not be reading or writing the
  context. Having moved it in, the caller *cannot*, and that is enforced by the
  checker that already exists rather than by a comment.
- **Forgetting to unregister becomes a compile error**, which is a real bug class
  in every C callback API. A dangling callback into freed context is the exact
  failure this is meant to prevent, and linearity prevents it at the source
  rather than at the trampoline.

**The fire-and-forget case**, where the library never gives the callback back,
does not get an exception. The registration is still linear, and a program that
means to abandon it says so with a terminal consumer that takes it and returns
nothing. "I am deliberately leaking this" is a thing worth having to write.

**The design is now written down** in [callbacks.md](callbacks.md), worked
against the code the way the separate compilation design was, with a step order
to build it in. Three things it settles and one it found:

- `uses CallbackAbi` is **dropped**. A `$handler` parameter with a function bound
  on an `extern fn` already says the extern wants a trampoline, and a capability
  that supplies nothing is a keyword pretending to be one.
- The handler's **first parameter is the context**, which is what makes the
  trampoline derivable, and the extern parameter of that same type is the one
  passed as the `void*`, so the declaration is written in the order C wants.
- The registration is `linear` and the context moves in and back out, which
  needs no new machinery: `check_ownership` and `check_linearity` already do it.
- **What it found:** the context has to name storage that outlives the
  registration, and a moved argument today is a place in the caller's frame.
  Nothing rejects that yet, because until now nothing in the language could keep
  a pointer past a call. That obligation is the difference between this and a
  prettier spelling of the C idiom.

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
