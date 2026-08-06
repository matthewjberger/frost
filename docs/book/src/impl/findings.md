# What the probes found

Questions asked of the compiler rather than of the design notes. Every claim
here is a program that was run; the programs are in the harness where they are
worth keeping.

## The region checker refuses one thing, and it is the right thing

The reported problem was that the checker rejects valid code when it cannot
trace provenance through deeply nested types. It does not. Six shapes, all
accepted:

| shape | written | answer |
| --- | --- | --- |
| one field | `b.room` | accepted |
| two fields | `o.held.room` | accepted |
| three fields | `c.b.a.room` | accepted |
| through an element | `b.slots[at].room` | accepted |
| from a branch | `if (pick) { return b.one }` then `b.two` | accepted |
| through a call | `inner(b)` where `inner` answers `b.room` | accepted |
| an array field | `d.a` where `a: [4]i64` | accepted |
| from the heap | `heap_slice($i64, 4)` | accepted |

Depth of nesting is not what the checker traces on. What it traces is where the
storage came from, and it has two answers it is happy with: a parameter, reached
through any number of fields, elements and calls, or an allocation.

The refusal class is one shape:

> a view whose storage is the calling function's own frame, on any path.

```frost
view :: fn() -> []i64 {
    var local : [4]i64 = [0; 4]
    local
}
```

> region: a pointer into the frame of 'view' is the call's answer; the storage
> it names dies when the call returns

That refusal is correct, and it holds for a function that answers with a frame
view on one path and a parameter view on another. Refusing the mixed case is
conservative, and it is the conservative direction that is sound: the answer is
a view of the frame on that path whatever the other path does.

No provenance annotation is designed, because no refusal was found that needed
one. Designing surface for a problem that does not reproduce would be surface
with no caller.

### Blast radius

The strongest form of the property holds, and is what the checker's shape
predicts. Adding a field to struct `S` cannot change the accept or reject status
of any function that names neither `S` nor a type containing `S`: provenance is
traced through the places a function names, and a struct it does not name is not
one of them.

```frost
Unrelated :: struct { x: i64, y: [8]f32 }   // y added
Box :: struct { room: []i64 }
Slot :: struct { held: Box }
view :: fn(s: Slot) -> []i64 { s.held.room }
```

Adding `y` changes nothing about `view`, and nothing about a function that
answers with a frame view. What a field addition does change is layout, which is
global and loud by design, and the type errors that follow name the field.

## Callee determinism holds for a name, not for a value

The thesis is that the text of a call site fully determines what it calls. Asked
of the compiler, it is true of a call to a name and false of a call to a value.
All four of these compile and run:

| shape | written | answers |
| --- | --- | --- |
| parameter | `run :: fn(f: fn(i64, i64) -> i64) -> i64 { f(2, 3) }` | 5 |
| local | `f := add` then `f(2, 3)` | 5 |
| reassigned local | `var f := add`, `f = mul`, `f(2, 3)` | 6 |
| struct field | `t.op(2, 3)` | 5 |

So the receiver-inference problem is not removed, it is relocated to dataflow: a
reader of `f(2, 3)` has to find what `f` was bound to, and with `var` that is a
question about every assignment that reaches the call.

The honest statement of the thesis is therefore scoped:

> A call whose callee is written as a name is fully determined by the text of
> the call site: there is one namespace, no overloading, no methods and no
> traits, so the name is the callee. A call whose callee is an expression names
> a value, and what that value is, is a dataflow question like any other.

The scope is worth what it costs. Forbidding function values entirely would make
the unscoped claim true and would take the callback tables, the render graph's
pass functions and the ECS's system lists with it. Restricting them to a binding
that is never reassigned would keep those and cost the `var f` case, which the
corpus does not use. That is the cheapest strengthening available and it is a
language change, so it is written down here rather than made.

Inside a generic body the same rule holds with one addition: a call to a `$f`
compile-time parameter is resolved where the generic is expanded, so it is
determined by the instantiation's text rather than the template's. `where`
bounds are preconditions the compiler already answers rather than a dispatch
mechanism, so nothing about them makes a call resolve differently per type.

## Allocation reaches the process through one door

The invariant behind "nothing allocates behind your back", stated precisely:

> Every allocation is a call to `frost_rt_heap_alloc`, `frost_rt_heap_realloc`
> or the arena, all of which are `extern fn` rather than `safe extern fn`, so
> reaching them takes an `unsafe` block. The only safe way through is a named
> function in `std/mem.frost`, and the only safe way to an arena is a `uses`
> capability written into the signature.

Verified by inspection: `std/mem.frost` is the one file in the tree that
declares those externs, and nothing else in the corpus names them. A function
that allocates therefore either calls something named in `std/mem.frost`, or
declares `uses Arena`, or has an `unsafe` block in it. All three are visible in
the text.

## The multi-return refusal was one build away

The reported problem was that the unsafe gate cannot name the type of a binding
introduced by a destructure, so `view, count := split(data)` followed by
`view[0]` is refused. Written that way, both compilers take it. The refusal
lives one build further on.

The gate reads what each function answers with off the top level, and a module
compiled on an earlier run comes back as a declaration built from its interface.
The list of values was read off the constant a source file spells, so the second
`--incremental` build of a program whose multi-return function lives in an
imported module bound every name in the destructure to no type, and the index
rule met a base it could not name:

> indexing a value whose type is not known here is unchecked, so it belongs in
> an `unsafe` block

The first build took the same program. A suite that compiles each program once
cannot see this, which is why the test builds three times. The self-hosted
compiler lowers the destructure as it parses, so the gate meets ordinary field
reads and it never had the hole. Held now by
`a_destructure_of_a_cached_module_is_taken_on_every_build`, with the shape
itself, over `[]T`, `str`, a struct holding a view and `[N]T`, in the
same-language table.

The sibling question, whether the two compilers agree about what a return type
list is, found two places where they did not. Each was the self-hosted compiler
letting the struct behind the list reach the surface:

| written | bootstrap | self-hosted, before |
| --- | --- | --- |
| `return { value0 = .. }` against `-> (i64, i64)` | refused | taken |
| `held := split(4096)` then `held.high` | refused | taken, and it ran |

Both are what
[5.2a of declarations.md](../reference/declarations.md) already said: a `return`
writes the names the signature gave, and a call answering with a list is bound
by a list of names. `value0` and `held.high` reach field names the compiler
picked for itself, and the two compilers pick different ones (`__multi_i64_i64`
against `__Multi0`), so a program written to either would have been written to
one compiler.

The first of those went away entirely rather than being fixed twice. No
signature in the corpus leaves a value unnamed, and the unnamed form's only
effect was to call the fields `value0` and `value1`, so a return type list now
names every value and the synthesis is gone from both parsers along with the
refusal that guarded it. The second is a refusal in both compilers under one
wording.

## `_` was an ordinary name in one compiler

Asking what the call site needed turned up a third divergence, and the reference
had already ruled on it: 2.3 says the lone underscore is a distinct token, the
wildcard, and never a binding name. The bootstrap has `Token::Underscore` and
enforces that. `selfhosted/lexer.frost` had no underscore token at all, so `_`
fell into the identifier rule and became an ordinary local:

```frost
high, _ := split(4096)
a, _    := split(512)      // shadows the first
print_int_line(high + a + _)   // printed 18 under the self-hosted compiler
```

`_ := 5` then reading `_` printed 5 there and was two separate errors under the
bootstrap. Neither compiler had a discard. One refused the spelling and the
other gave it semantics nobody wants.

Both now lex it as its own token, and a binding list takes it as a value the
caller has no use for, in any position and any number of times. The field is
still read, into storage named `__discard<n>` that no source can spell, so a
linear value taken by a `_` is still a resource somebody owes a consumer rather
than one dropped silently. `_ := 5` and `_` in expression position are refused
by both, word for word.

## An address read as something other than a number

Following the same question one step further found a fourth. The self-hosted
`types_compatible` answers a chain of rules and ends in `true`, so a pair
neither side had a rule for was taken. Asked over the scalars:

| written, where `p` is a `^i64` | bootstrap | self-hosted, before |
| --- | --- | --- |
| `q : i64 = p` | taken | taken |
| `back : ^i64 = q` | taken | taken |
| `f : f64 = p` | refused | taken |
| `b : bool = p` | refused | taken |

The integer directions are right and both had them: an address is a whole
number, which is what a call into C hands over and what address arithmetic
reads back. The other two are the fall-through. A float holds a different
encoding of the same bits and a `bool` holds one of two values, so neither is
an address read back.

The self-hosted compiler names the pair now instead of running out of rules.
The wording stays each compiler's own, since the bootstrap catches it in the
typed IR and names the IR local while the self-hosted one names the binding.
`both_compilers_refuse_an_address_read_as_a_float_or_a_bool` holds both
refusals and the two legal directions beside them, so it cannot be satisfied by
refusing every pointer conversion.

## A return type list could not carry a resource

Asking what a caller may leave unbound found the last one, and it was in both
compilers. A struct holding a resource is a resource, which is right, and the
struct a return type list becomes inherited it:

```frost
File :: linear struct { handle: i64 }
pair :: fn(n: i64) -> (opened: File, count: i64) { ... }

held, count := pair(3)
close(held)              // consumed, correctly
```

> bootstrap: `linearity: linear value '__multi_result0' is not consumed on every
> path before return`
> self-hosted: `linear value '__multi114' is never consumed`

Both refused the correct program, and both named a temporary the compiler
invented. The lowering binds the call to that temporary and reads one field out
of it per name, and nothing treated the reads as emptying it, so the container
was a resource nobody could consume. A return type list carrying a `linear`
value was unusable.

The struct a list becomes is the one aggregate a program cannot hold. It is
built at the `return`, taken apart at the binding that reads it, and every field
is read exactly once on the way, so what it owes is exactly what its fields owe
and each of those lands on a name the binding introduced. Both closures leave it
out now, and the obligation lands where a reader can see it.

The other half is the `_` added the same day. A resource taken by one is refused
where the `_` is written:

```
this `_` drops a 'File', which is consumed exactly once; bind it to a name and
consume it
```

Said there rather than by the linearity walk, which runs on what the lowering
left behind and would have named `__discard0`. A diagnostic that names a name
nothing in the program spells is a diagnostic a reader cannot act on, which is
what both compilers were doing here before.

## Reading the day's own work back

Seven faults, found by reading every changed file rather than by a failing test.
Two are worth keeping here because the shape recurs.

**A name is a name.** The bootstrap told the struct a return type list becomes
apart from a declared one by its `__multi` prefix, and the self-hosted compiler
by a flag it set when it made one. So a program declaring `__multiHolder` fell
out of the bootstrap's linear closure, stopped being a resource, and leaked the
`File` in it while the self-hosted compiler refused the same program. The
lowering records the structs it makes now, in the `Ast` that every pass already
carries, so the two ask the same question the same way.

**A `str` is a `[]u8`, and only one compiler believed it here.** `slice_len` on
a `str` was refused by the bootstrap and taken by the self-hosted one, which
holds the two as one type. 3.2 says what the answer is, so the bootstrap was
behind. This one turned up in a probe written to confirm a different bug, which
is the argument for writing the probe rather than reasoning the bug closed: the
program you write to check one thing runs everything around it too.

What reading missed: an offset this chapter would have claimed was zero, on the
strength of every `Call` node being built with `c = 0`. Running it showed the
diagnostic landing on the call, correctly. The claim was wrong and only a
program said so.

## An array parameter is sliced where the caller holds it

An array coerces to a slice of the whole array, and a parameter of array type is
a borrow, so the slice views the caller's storage. Six positions take a slice: a
binding with an annotation, an assignment, a field of a literal, an argument, a
`return`, and the same through a read borrow rather than a `mut`. Every one of
them was wrong somewhere.

The self-hosted compiler refused three of the six, and for the other three wrote
the bare address where a slice belongs, so what read the length read whatever
sat beside it: `view[3]` on a four-element array reported a length of two, and
two of the shapes segfaulted. The bootstrap took all six and copied the array
into the callee's frame first, so a write through the slice landed in the copy
and a slice handed back pointed into a frame that was gone. Cranelift printed
the caller's value anyway, because the dead frame still held it; the C backend
printed 8.

The program that separates them is one line:

```frost
bump :: fn(mut source: [4]i64) {
    view : []i64 = source
    view[0] = 99
}
```

The write reaches the caller's array, which is what `mut` means everywhere else
in the language. All five backends agree on all six positions now, and the
program above is in the same-language table under
`an_array_parameter_is_sliced_where_the_caller_holds_it`.

## A container that could not say which slots were filled

`columns<T, N>` had two access forms and neither answered the question a system
asks most: which slots hold an element. `c.field` is the whole `[N]` array,
released slots included, unchecked. `c[handle].field` is one slot, checked. So
the shortest loop to write, and the one a model writes from a spec, walks the
capacity and reads storage nobody filled.

The check exists per handle. The fast path was defined as the path around it.

Reading the container to design a live walk turned up the sharper version: **the
correct loop could not be written at all.** `columns_insert` does not touch
`generations` and `columns_release` bumps it, so a slot at generation zero is one
that was never filled as much as one that is live. Nothing distinguished them.
Liveness existed in exactly one place, `free_list[0..free_count]`, which is in
release order and would cost a scan per slot to consult. So a live walk needed
new bookkeeping before it needed syntax.

`live_words` is one bit per slot in slot order, which is the order a column is
stored in. A dense list of live indices would also give a branchless walk, but it
scrambles the order into a gather, and striding a column is the entire reason the
container is structure-of-arrays. `N/8` bytes against `16N` for a dense list and
its inverse.

The surface is two characters longer than the wrong loop:

```frost
for slot in 0..N     { ... }
for slot in live_slots(c)  { ... }
```

and it drops the requirement that `N` be in scope, which inside a generic is a
`$N` parameter threaded only to bound a loop.

### The dense fast path was not worth its code

The first design split the walk in two: a word of all ones ran a sixty-four trip
counted loop with no bit arithmetic, and a partial word walked set bits. It was
meant to make the common case the loop a hand-written packed walk emits.

It costs a second copy of the body at every walk. And the vectorization it was
supposed to enable does not happen: the body indexes columns with a value the
compiler would have to prove affine through a bitset, which neither Cranelift nor
the C backend does. What it actually bought was two ALU operations per element,
neither of which branches.

One loop shape now, in both compilers. Dead slots are still skipped sixty-four
at a time and no element is asked whether it is live.

### `break` decided the shape of the self-hosted walk

The bootstrap builds the walk out of blocks, so `break` jumps to the exit block
and leaves the whole thing. The self-hosted compiler writes a `for` out as source
in the parser, where a `break` inside the inner of two loops leaves only that
one, and there is no labelled break. A word loop holding a bit loop would have
made `break` in the reader's body continue to the next word.

So the self-hosted walk is one loop with the refill as a branch inside it. It
pays a compare per element that is taken once in sixty-four, and `break` and
`continue` mean what they mean anywhere else. Both compilers pass the same
program in the same-language table, which is where that had to be settled.

## A handle said which slot, never which container

A `Handle<T>` carried a slot index and a generation. The generation catches a
handle to a slot that has been released and refilled. It says nothing about which
container the slot is in.

Two containers of the same element type and the same capacity are the ordinary
shape: `active` and `pending`, `current` and `next`, front and back. A handle from
one used against the other has an index in range on both, and the generations
match whenever the two slots have been released the same number of times.
Differing capacities turn most such uses into a bounds abort, so the exposure was
narrowest exactly where it is most likely to be written.

The framing that decided it was reading `slab_reset`: it writes zero to every
generation. So right after two containers are reset, **every** cross-container
handle validates, on every slot. Not a coincidence in the tail, and not
probabilistic: a guaranteed false accept, in the state every program starts in.

The fix is seven bits taken from the generation and stamped into every entry of
`generations` at reset, from a counter in the runtime. It costs:

- **nothing per handle.** Still an `i64`, still converts freely.
- **nothing per deref.** The number sits in the same word as the generation and
  the deref already compares that word. A separate field on the container would
  have cost a load, a shift and an or on every read.
- **nothing in the layout.** A container stores no handles, only `generations`.
  `columns<T, N>` is what it was.
- **generation range**, from 2^31 to 2^24 releases of one slot. Sixteen million,
  or seventy-seven hours at frame rate on a single slot, and the slot is retired
  rather than corrupted when it gets there.

Seven bits rather than eight because a packed handle has to stay positive: the
generation is read back sign-extended, and an eighth bit would put a one in bit
63 of the handle and make the comparison fail against its own container.

What it does not catch: two containers that draw the same number, one time in a
hundred and twenty-seven; a container reset in a loop, which comes back round
after that many; and the `slab_slot` escape hatch, where a caller asks for a raw
index once and indexes storage with it, which is checked by nothing by design.

## An array's length had to become arithmetic

`live_words` on a `columns<T, N>` is synthesized, so its length is worked out in
the compiler. `Slab<T, N>` is library Frost, and a length could only be a number
or a name standing for one. `[(N + 63) / 64]i64` is not writable that way, so the
slab could not carry the same record and the two containers would have differed
in what they guarantee. The reference says switching one for the other changes
the container token and the prefix and nothing else; that sentence would have
become false.

So a length is arithmetic now: `+ - * / %` and brackets over numbers, module
constants, and the size parameters a generic binds. Nothing else. No call, no
comparison, no name that is not a size, which keeps it inside the bound every
compile-time construct in the language has: what it costs is decided by how it
was written.

The two compilers hold it differently and that is the same language read twice.
The bootstrap parses a length into a small tree and carries it in
`Type::ArrayGeneric` until the generic that binds its names is instantiated,
because a template's body is lowered once. The self-hosted compiler parses a
generic body once per instantiation with the parameters already bound, so every
name is a number by the time it reads one and the answer is a number too.

### Adding a field to a library struct broke thirty-two literals

Every `Slab { storage = ..., generations = ..., free_list = ..., free_count = 0 }`
in the tree stopped compiling, and the fix could not be mechanical: the new
array's length is `(N + 63) / 64`, a number the writer would have to work out per
capacity.

`slab_new()` is the answer, and it is the twin of the `columns_new()` that
already existed for the same reason. A zeroed container of the type the context
wants, then `slab_reset`. Every literal in the tree is one call now, which is
what they should have been: enumerating four arrays to write zeros into them was
the worst part of using a slab.

### One divergence found on the way

`slice_len` of a fixed array is its length, which the type already carries. The
self-hosted compiler answered it, emitting the constant; the bootstrap refused
with "expected a slice value". A `str` is a `[]u8` and an array coerces to a
slice everywhere else, so the self-hosted side was right, and the bootstrap
answers it now.

## Asking a node its type is not free

The unused-result rule needs to know whether a statement answers with a failure.
The obvious way is to ask the call node its type, and it is wrong: that walk
instantiates a generic to work the answer out, so a check running over every
statement builds types nothing else ever wanted. `std/ecs.frost` ran an arena
out of room compiling under it.

Read off what the callee was declared to answer with instead. It is a table
lookup, it needs no instantiation, and for this question it is the same answer.

The general shape, for the checks still to be written: a walk that visits
everything must not call anything that can *create*. Recording a fact where it
is made and reading it back is the pattern, the same one the multiple-return
structs and the failure-set enums already use for a different reason.

The diagnostic was its own defect. "an arena of 4096 ran out of room" named
neither the arena nor the cause, and a dozen arenas are made at that size, so
the number named every one of them and none of them. It names the element type
now, which is what tells them apart.

## A word that is reserved is a word a program cannot use

`packed` and `align` went in as keywords, and `std/slab.frost` stopped
compiling at line 98:

```frost
packed := (s.generations[index] << 32) | index
```

The name is the right one for what it holds, and the layout feature has no
claim on it. Both words are contextual now, read by the shape after them: the
`struct` after `packed` and the `(` after `align`. Nothing else in the language
follows a name with either.

That is the fourth word to arrive as a keyword and leave as one: `flags`,
`value`, `test` and `export` each read this way already. The rule the tree now
holds to is that a marker word is contextual unless it opens a statement, since
a marker sits in a position an expression cannot reach and the token after it
says which it is.

`tests/editor_grammar.rs` cross-checks the highlighter against
`frost::KEYWORD_NAMES`, so a word that leaves the keyword table has to join the
`CONTEXTUAL` list there or the grammar reads as drifted. That test is what
catches the half-finished version of this change.

## Four backends compute a layout and three of them by hand

A stated layout has to reach every emitter, and they get there differently. The
bootstrap lays every struct out in `try_struct_layout` and its C emitter writes
the offsets it worked out, so nothing about the C reveals that a struct was
packed. The self-hosted compiler lays out for its assembly backend in
`selfhosted/layout.frost` and hands its C backend a real `struct` declaration,
so that one needs `__attribute__((packed))` and `__attribute__((aligned(N)))`
to say the same thing to the C compiler.

Three separate implementations of one rule is where a divergence hides, and one
did: `sizeof` agreed on all four while `offset_of` did not. The field walk that
`for field in fields(T)` unrolls computes offsets of its own, in
`unroll_fields`, rather than asking the layout code, so it kept answering the
natural offsets after the layout code learned about packing. A grid of six
structs printing `sizeof` and every `offset_of` through all four backends is
what found it; a probe that printed sizes alone would have passed.

## A constant is read before any body is, and both compilers had to work around it

A call in a constant or an array length has to answer before the parse that
needs the number gets there, and at that point no function body has been
parsed. Neither compiler could reach for its own tree, and each went the other
way round it.

The bootstrap already scanned the tokens for constants before parsing, since an
array size is part of a type. So the callee is parsed there too, out of its own
token range, into an `Ast` of its own that nothing else ever sees. Reusing the
ordinary expression parser on a slice of tokens is what made that cheap: a
function literal is an expression, so `parse_expression` on
`fn(a: i64) -> i64 { ... }` gives the body back. The one catch is that a
function whose parameters carry types parses as `Proc` rather than `Function`,
and a check for one of the two silently found no bodies at all.

The self-hosted compiler holds every module in one token stream and reads
constants in a first pass over it, so the callee's tokens are already there. It
walks them directly, with a precedence chain level for level with the parser's.
Parsing the body into nodes instead was the obvious idea and it is wrong: the
self-hosted parser interns a global and notes a type-argument tuple as it reads
a call, so a body read early would reorder tables the emitted output depends on.

**The argument that was typed by a signature nobody had read.** The first
attempt in the self-hosted compiler let the ordinary parser read the constant's
value and folded the resulting tree. A call's arguments are read against the
callee's declared parameter types, and at constant time the callee has no
declared anything, so the literal `300` was typed by whatever the empty entry
held and came out as a node the folder did not recognize. The value is read off
the tokens now, and the parse of it never happens.

**A fault that moves the cursor backwards loops forever.** A compile-time call
sets the cursor to the callee's body and puts it back when it answers. A fault
leaves through the escape rather than through the putting back, so the cursor
stayed inside the callee. Recovery then synchronized from there, found the same
declaration again, and reported one fault 262,000 times. Both recovery turns
carry the cursor they began at now and only ever move it forward. The bug is
older than the feature: any pass that moves the cursor and then faults had it.

## The two compilers vectorize at opposite ends of the pipeline

Elementwise arithmetic over a fixed array of numbers is one language rule and
two implementations that share nothing.

The bootstrap writes the lanes out in its IR: a fresh local of the vector type,
then a load, an operation and a store per lane. Every one of its three backends
then carries it with the code it already had, and the C compiler folds the run
back into `mulps` and `addps` at `-O2`, which is what a check of the emitted
assembly shows.

The self-hosted compiler splits on the element type. A vector of whole numbers
becomes an array literal of lane expressions at parse time, because an overflow
aborts and says where, and a packed add has no way to say which lane it was. A
vector of floats stays one `Bin` node, and the assembly backend emits `movups`
and the packed arithmetic sixteen bytes at a time. That split is why the
assembler grew `addps`, `subps`, `mulps`, `divps` and their `pd` forms: eight
mnemonics whose encoding differs from the scalar ones only in the prefix.

### A parse-time question needs a parse-time answer

`type_of` reads the local table the *walks* build, and during the parse that
table is empty: every name answers `i64`. The parser keeps its own table, and
what the vector rules ask has to go to that one. Three separate places had to
learn: a name, an index of one, and an operator over two of them.

The failure is quiet. A vector operation whose operands read as `i64` is an
integer add, and the program compiles and prints nonsense. It showed up as
`6` becoming `1.6352e+12`, and the same shape came back twice more, once for a
nested expression and once for a parameter, which arrives as a borrow and so is
not an array until the borrow is read through.

### A diagnostic that names a type is a diagnostic the two can differ on

The first wording was `'{type}' and '{type}' do not go together`, and the two
compilers render a fixed array differently: `[4]f32` in the bootstrap and
`[]f32` in the self-hosted, whose renderer does not print a length. The
diagnostic says `a vector of 4 f32` now, built from the length and the element
rather than handed to a type renderer. A sentence a reader sees is one sentence
in one language, so where the two would render it differently, neither renders
it.
