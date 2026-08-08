# The entity-component system

`std/ecs.frost` is an archetype ECS: entities holding the same set of components
share a table, a table keeps one contiguous column per component, and a system
walks columns rather than chasing entities. It is ordinary Frost you can read,
copy, or replace.

Everything else follows from one decision: a component is plain data. No
destructor, no copy constructor, nothing a type knows that the compiler does
not. Every operation a column performs is a move of `item_size` bytes, whether
it is growing, removing a row, or carrying a row to another table, so there is
no table of per-type function pointers to carry. The registry holds a size per
component and nothing else.

```frost
import "ecs.frost"

Position :: struct { x: f32, y: f32 }
Velocity :: struct { x: f32, y: f32 }

main :: fn() -> i64 {
    var world := ecs_new()
    position := ecs_register($Position, world)
    velocity := ecs_register($Velocity, world)

    ship := ecs_spawn(world)
    ecs_add($Position, world, ship, position, Position { x = 0.0, y = 0.0 })
    ecs_add($Velocity, world, ship, velocity, Velocity { x = 1.0, y = 0.5 })

    var q := query_begin(world)
    query_with(q, position)
    query_with(q, velocity)
    while (query_next(world, q)) {
        var p := query_column($Position, world, q, position)
        v := query_column($Velocity, world, q, velocity)
        var i : i64 = 0
        while (i < q.count) {
            p[i].x = p[i].x + v[i].x
            p[i].y = p[i].y + v[i].y
            i = i + 1
        }
    }
    ecs_free(world)
    0
}
```

## Entities

An entity is an id and a generation. The id indexes the slot table. The
generation is bumped when the id is freed, so a handle kept past a despawn names
a generation the slot no longer has and every lookup refuses it. This is the
rule `Handle<T>` follows for a pool, written out again here because an entity's
rows live in whichever table matches its current component set.

| Call | What it does |
| --- | --- |
| `ecs_spawn(world) -> Entity` | A live entity with no components |
| `ecs_despawn(world, entity)` | Frees the id and invalidates every handle to it |
| `ecs_alive(world, entity) -> bool` | Whether this handle names the entity holding its id now |
| `ecs_count(world) -> i64` | How many are live |

## Components

A component is named by its index, handed out by `ecs_register` in registration
order. A table's mask is the set of indices it holds, its columns are in
ascending index order, and the column for a component is found by counting the
bits below it, so nothing maps a component to a column while the program runs. A
world holds up to 256 component types, one per bit of a four-word mask.

`ecs_register` also records which type the component was registered under, so a
query can name one by writing its type. `component_of($T, world)` answers that
index, and `for_each` takes a list of types on the same footing.

| Call | What it does |
| --- | --- |
| `ecs_register($T, world) -> i64` | Registers `T` and answers its index |
| `ecs_add($T, world, entity, component, value)` | Gives it the component, migrating the entity if it is new |
| `ecs_remove(world, entity, component)` | Takes it away, migrating the entity |
| `ecs_has(world, entity, component) -> bool` | Whether it holds it |
| `ecs_get($T, world, entity, component) -> T` | The value, by copy |
| `ecs_set($T, world, entity, component, value)` | Overwrites it, stamping the row |
| `ecs_slice($T, world, table, component) -> []T` | One table's column, to write through |

Adding or removing a component moves the entity to the table for its new set,
carrying every column the two tables share. A despawn moves the last row into
the hole it left and tells whichever entity moved where it went, which keeps
every column contiguous.

## Queries

A query is a cursor over the tables holding a set of components:

```frost,sketch
var q := query_begin(world)
query_with(q, position)
query_with(q, velocity)
while (query_next(world, q)) {
    var p := query_column($Position, world, q, position)
    v := query_column($Velocity, world, q, velocity)
    entities := query_entities(world, q)
    ...
}
```

Written this way a query has no arity limit and captures nothing. The body sits
where you write it, so what it reads is the enclosing function's own locals.
`for_each` is the same walk with the body handed in as a compile-time argument,
for a system short enough that the cursor is the longer half. Its components are
a compile-time list of types, so there is no arity to name and no limit. The
list drives the mask a table is matched against and the column each element
reads through, and a `for` over it unrolls both where the call is written.

```frost,sketch
integrate :: fn(mut p: []Position, mut v: []Velocity, count: i64) {
    var i : i64 = 0
    while (i < count) {
        p[i].x = p[i].x + v[i].x
        i = i + 1
    }
}

for_each($integrate, world, no_filters(), $Position, $Velocity)
```

A body taking a fourth component is a fourth element and a fourth parameter.

## Filters

What a query asks for beyond the components its body reads:

```frost,sketch
var f := no_filters()
f = filter_without(f, frozen)             // table level
f = filter_changed(f, velocity, last_run) // row level
for_each_row($move, world, f, $Position, $Velocity)
```

`without` is matched against a table's mask, so a whole archetype is skipped
before any of its rows is touched. `changed` and `added` read the per-row ticks
a column already carries, so the table is still walked and what they save is the
body. A filter goes through the call that matches its level.

`for_each` hands the body whole columns and a row count, which is the faster
form and the one to reach for. `for_each_row` hands it one row at a time, which
the row-level filters need, since `changed` is a question about a row.

The `$body` argument folds to a direct call, so the sugar costs nothing over the
cursor form.

## Change detection

Every row carries two ticks: when it was last written, and when the entity
holding it was first given the component. The world's clock is advanced by the
program, once a frame, so everything written during a frame shares one time.

```frost,sketch
watermark := ecs_tick(world)
... systems run ...

var q := query_begin(world)
query_with(q, transform)
while (query_next(world, q)) {
    stamps := query_changed(world, q, transform)
    t := query_column($Transform, world, q, transform)
    var i : i64 = 0
    while (i < q.count) {
        if (stamps[i] >= watermark) { upload(t[i]) }
        i = i + 1
    }
}
```

The ticks are data, so you write the test where the decision is, and a body that
wants both the changed and the unchanged rows has both. `ecs_changed_since` and
`ecs_added_since` ask the same question about one entity. A migration carries a
row's ticks with it, so gaining a different component leaves the ones already
there reading as unwritten.

## Resources

A resource is a value the whole world shares: the renderer's device, the input
state, the frame's time. It is a component with one row and no entity, stored as
a column of one and reached through the same typed slice.

```frost,sketch
time := ecs_resource_register($Time, world)
ecs_resource_set($Time, world, time, time_new())
held := ecs_resource($Time, world, time)
var place := ecs_resource_slice($Time, world, time)
place[0].frame = place[0].frame + 1
```

Setting a resource moves the value into the world. A resource that has to be
released is the world's from there, and the name that held it is free of it.

Reading one back has two shapes. `ecs_resource` answers with the value, for a
caller taking it back to release it. What comes back owns what it holds, so a
system that only meant to look at a `Hierarchy` would be left with vectors to
free. `ecs_resource_ref` answers with a borrow, reading the resource where it
lives and owning nothing.

```frost,sketch
tree := ecs_resource_register($Hierarchy, world)
ecs_resource_set($Hierarchy, world, tree, hierarchy_new())

held := ecs_resource_ref($Hierarchy, world, tree)   // look at it
print("{}\n", hierarchy_child_count(held, parent))

giving := ecs_resource($Hierarchy, world, tree)     // take it back
hierarchy_free(giving)
```

A system is a `fn(mut World)` and captures nothing, so everything it works on is
a component or a resource. Reach a resource holding something the world is
responsible for through the borrowing read.

## Events

A channel one system writes and another reads. Events live in a column, so
sending one is a push of bytes and reading them is a slice. A reader keeps its
own place by sequence number, so clearing the channel leaves every reader on
exactly the events it has yet to see:

```frost,sketch
var damage := events_new($Damage)
var renderer := reader_new()
events_send($Damage, damage, Damage { amount = 3 })

held := events_read($Damage, damage, renderer)   // what this reader has not seen
events_clear(damage)                             // drop the frame's events
```

A reader that fell behind a clear is caught up to the start of what is left.

## Tags

A tag marks an entity without costing a mask bit or moving a row, so a loop can
flip one on and off at no migration cost. The generation is stored beside the
mark, so a tag left on a despawned id stays off the entity that gets that id
next.

```frost,sketch
var selected := tag_new()
tag_add(selected, entity)
if (tag_has(selected, entity)) { ... }
```

## Commands

A structural change made while a query is walking would move the rows the walk
is holding, so it is queued and applied when the walk is done:

```frost,sketch
var queued := commands_new()
... during the walk ...
commands_despawn(queued, entities[i])
... after it ...
commands_apply(queued, world)
```

## Hierarchy

A parent-child relation held beside the world, since it is a relation between
two entities. Three arrays indexed by entity id give a tree walked with no
allocation per node: the parent, the first child, and the next sibling.

```frost,sketch
var tree := hierarchy_new()
hierarchy_attach(tree, wheel, car)
var child := hierarchy_first_child(tree, car)
while (is_no_entity(child) == false) {
    ...
    child = hierarchy_next_sibling(tree, child)
}
hierarchy_despawn_tree(tree, world, car)   // the car and everything under it
```

## Schedules, states and time

A system is a function of the world. A schedule is a list of them with a stage
each, run in ascending stage order, so ordering is a number:

```frost,sketch
var frame := schedule_new()
schedule_add(frame, Stage::First, read_input)
schedule_add(frame, Stage::Update, integrate)
schedule_add_in_state(frame, Stage::Update, PAUSED, draw_menu)
schedule_add(frame, Stage::Last, upload)

schedule_run(frame, world, states_current(states))
```

The system is a function pointer, since a schedule is built while the program
runs. `for_each` and `for_each_row` cover the other half, the inner loop where
the call has to fold away.

A state change is requested during a frame and taken between frames, so a system
that asks to leave a state keeps the schedule it started under until it
finishes. `Time` carries the frame number, the last delta, and the total
elapsed.

## The structural log

Off until a program asks for it. With `ecs_log_enable(world, true)` the world
records each spawn, despawn, add and remove with the tick it happened at, as a
`ChangeKind` beside the entity and the component's mask. A save file writing a
delta, or an editor keeping a list in step, reads that log.

## The unsafe floor

The unsafe floor is the column: raw bytes with a width. Everything above it
reaches an element through `column_of`, which hands out a bounds-checked `[]T`,
and through generational handles that refuse a stale entity. A program using the
ECS writes no `unsafe` of its own.

## Tests

Every part above has a `test` block beside it in `std/ecs.frost`:

```
frost --test std/ecs.frost
```

They run identically under the bootstrap compiler and under the self-hosted one
on both of its backends.
