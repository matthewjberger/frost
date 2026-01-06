# Frost Memory Model

## Design Goals

1. Memory safety without lifetime annotations
2. Explicit allocators (Odin/Zig style)
3. Simple mental model
4. Game engine optimized

## Type Categories

### 1. Value Types (Copy)

Primitives that are copied on assignment:

```
i8, i16, i32, i64
u8, u16, u32, u64
f32, f64
bool
```

Also copy: raw pointers `^T`, handles `Handle<T>`, function pointers.

### 2. Owned Types (Move)

Types that have a single owner and move on assignment:

```
str                    // Owned string
[]T                    // Owned dynamic array
[N]T                   // Owned fixed array (if T is owned)
StructName             // Owned struct (if any field is owned)
EnumName               // Owned enum (if any variant has owned fields)
```

### 3. References (Second-Class)

References are **second-class citizens**:
- Can exist as local variables
- Can be function parameters
- **Cannot** be stored in structs
- **Cannot** be returned from functions
- **Cannot** be stored in arrays

```frost
// ALLOWED
update :: fn(pos: &mut Position) { ... }
ref := &value;
process(ref);

// FORBIDDEN
CachedRef :: struct {
    pos: &Position,     // ERROR: cannot store reference in struct
}

get_ref :: fn(world: &World) -> &Position {  // ERROR: cannot return reference
    ...
}

refs := [&a, &b, &c];   // ERROR: cannot store references in array
```

**Rationale:** If references cannot be stored or returned, they cannot outlive their referent. The compiler only needs scope-based checking.

### 4. Arenas (Region-Based)

Arenas provide bulk allocation with region tracking:

```frost
Arena :: builtin_type {
    new :: fn(size: usize) -> Arena
    alloc :: fn<T>(self: &Arena, value: T) -> ^T
    reset :: fn(self: &mut Arena)
}
```

Arena-allocated pointers are tagged with their arena:

```frost
frame : Arena = Arena::new(megabytes(4));

// ptr has type: ^[frame]Position
// The [frame] tag tracks which arena owns this memory
ptr := frame.alloc(Position { x = 0.0, y = 0.0 });
```

**Region rules:**
- `^[arena]T` cannot be stored in a struct that outlives `arena`
- `^[arena]T` cannot be returned from a function if `arena` is local
- Arenas can be passed to functions; their pointers valid for that call

```frost
process_frame :: fn(frame: &Arena) {
    temp := frame.alloc(Data { ... });
    // temp valid here
}
// temp no longer accessible (scope ended)

// FORBIDDEN
global_ptr : ^Position;  // Untagged pointer - must use unsafe

store_ptr :: fn(frame: &Arena) -> ^Position {
    frame.alloc(Position { ... })  // ERROR: cannot return arena pointer
}
```

### 5. Handles (Generational Indices)

For persistent references to pooled objects:

```frost
Handle :: builtin_type<T> {
    index: u32,
    generation: u32,
}

Pool :: builtin_type<T> {
    new :: fn(capacity: usize) -> Pool<T>
    alloc :: fn(self: &mut Pool<T>, value: T) -> Handle<T>
    get :: fn(self: &Pool<T>, handle: Handle<T>) -> ?&T
    get_mut :: fn(self: &mut Pool<T>, handle: Handle<T>) -> ?&mut T
    free :: fn(self: &mut Pool<T>, handle: Handle<T>)
}
```

Usage:
```frost
entities : Pool<Entity> = Pool::new(1024);

player := entities.alloc(Entity { health = 100, ... });
enemy := entities.alloc(Entity { health = 50, ... });

// Safe access - returns null if handle is stale
if let Some(e) = entities.get(player) {
    print(e.health);
}

entities.free(enemy);
entities.get(enemy);  // Returns null (generation mismatch)
```

**Properties:**
- Handles are Copy (just two u32s)
- Can be stored in structs
- Can be returned from functions
- Access is O(1) with one branch (generation check)
- Dangling handles return null, never crash

### 6. Unsafe Blocks

For FFI, intrinsics, and when you need raw pointers:

```frost
unsafe {
    raw_ptr : ^Position = ptr_cast(some_handle);
    raw_ptr.x = 1.0;  // No safety checks

    // Can store references (at your own risk)
    stored_ref = &value;

    // Can call C functions
    c_malloc(1024);
}
```

Inside `unsafe`:
- Raw pointers `^T` without arena tags
- Can store references in structs
- Can return references from functions
- Can call extern functions
- No bounds checking on array access

## Reference Rules Summary

| Operation | `&T` / `&mut T` | `^[arena]T` | `Handle<T>` | `unsafe ^T` |
|-----------|-----------------|-------------|-------------|-------------|
| Local variable | Yes | Yes | Yes | Yes |
| Function param | Yes | Yes | Yes | Yes |
| Store in struct | **No** | **No** | Yes | Yes |
| Return from fn | **No** | **No** | Yes | Yes |
| Store in array | **No** | **No** | Yes | Yes |
| Outlive scope | **No** | **No** | Yes | Yes |

## Borrow Rules

Within a scope:
- Multiple `&T` allowed
- One `&mut T` exclusive
- Cannot use owned value while borrowed

```frost
value := 100;
r1 := &value;
r2 := &value;      // OK: multiple immutable
print(r1^ + r2^);

mut x := 50;
m := &mut x;
// r := &x;        // ERROR: already mutably borrowed
m^ = 200;
```

## Memory Management Patterns

### Pattern 1: Frame Allocator (Per-Frame Scratch)

```frost
game_loop :: fn() {
    frame : Arena = Arena::new(megabytes(4));

    while running {
        defer frame.reset();

        // All frame allocations freed at once
        particles := frame.alloc([Particle; 1000]);
        process_particles(particles);
    }
}
```

### Pattern 2: Pool for Game Objects

```frost
World :: struct {
    entities: Pool<Entity>,
    positions: Pool<Position>,
    velocities: Pool<Velocity>,
}

spawn_enemy :: fn(world: &mut World) -> Handle<Entity> {
    pos := world.positions.alloc(Position { x = 0.0, y = 0.0 });
    vel := world.velocities.alloc(Velocity { dx = 1.0, dy = 0.0 });
    world.entities.alloc(Entity { position = pos, velocity = vel })
}
```

### Pattern 3: Level Arena (Bulk Load/Unload)

```frost
Level :: struct {
    arena: Arena,
    meshes: []^[self.arena]Mesh,
    textures: []^[self.arena]Texture,
}

load_level :: fn(path: str) -> Level {
    level := Level { arena = Arena::new(megabytes(64)), ... };
    // All level data allocated in level.arena
    level
}

unload_level :: fn(level: Level) {
    // Single reset frees everything
    level.arena.reset();
}
```

