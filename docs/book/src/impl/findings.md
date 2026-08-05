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
