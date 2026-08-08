# Graphics

`lib/` sits outside `std/`: two C libraries bound to Frost and the pieces a
drawing program is built out of, with the programs that use them under
`examples/graphics/`. The bindings are how a Frost program talks to a real
graphics API, and the math they use comes straight out of [math.md](math.md).

## Four layers, reaching one way

Each layer may name the ones below it and none may name the ones above.

**`lib/platform/`** is the machine: a window, what was pressed, and the clock.

| File | What it is |
| --- | --- |
| `sdl.frost` | SDL3: a window, its events, and the clock that paces them. Hand-written |
| `platform.frost` | The window, the keyboard and the frame clock under one handle |

**`lib/renderer/`** is the device, and the two components it asks of an entity.

| File | What it is |
| --- | --- |
| `wgpu.frost` | The whole WebGPU API. Generated |
| `gpu.frost` | Bringing a device up, with or without a window to draw at |
| `renderer.frost` | The device, the surface and the frame under one handle |
| `frame.frost` | The frame the surface hands over, and where a pass draws into |
| `graph.frost` | A render graph: passes declare their targets, the order follows |
| `mesh.frost` | Geometry on the device, and the cache a program names it through |
| `geometry.frost` | Every mesh in one pair of buffers, and a mesh as a span of them |
| `material.frost` | A registry of surfaces, so a thing that is drawn carries a number |
| `texture.frost` | Images on the device, and render targets to draw into |
| `uniform.frost` | A uniform buffer, the layout that describes it and the group that binds it |
| `cluster.frost` | The lights of a frame, and the grid that says which of them reach where |
| `lit.frost` | The pass that draws a run of things, lit, with no shader to write |
| `render_world.frost` | What the renderer needs of an entity, and the run a pass walks |

**`lib/engine/`** works out where everything ended up and hands that over.

| File | What it is |
| --- | --- |
| `app.frost` | The window, the device, the world and the tables, opened once and given back once |
| `world.frost` | Transforms, a tree and the schedule that resolves them |
| `camera.frost` | Where the eye is and what it looks at |
| `light.frost` | A light in the world, and what the renderer reads instead of it |
| `scene_sync.frost` | The walk from a world to the flat run a pass draws |
| `gltf.frost` | A binary glTF file, read into geometry, materials and nodes |
| `log.frost` | Lines a program writes as it runs, kept in a file |

**`examples/graphics/`** are the programs.

| File | What it is |
| --- | --- |
| `window.frost` | Opens a window and pumps it until it is closed |
| `input.frost` | Drives the platform layer for a few seconds and reports |
| `triangle.frost` | Draws a rotating triangle into that window |
| `scene.frost` | Entities in an ECS, two passes, depth deciding what is in front |
| `spinning.frost` | Lit surfaces: a mesh cache, a material registry, two bind groups |
| `textured.frost` | The same field with its surfaces read off an image |
| `shadowed.frost` | An ECS schedule driving compute, shadows, a bloom chain and a second view |
| `gltf_model.frost` | A model read out of a file and spawned into the world |
| `lit.frost` | A world drawn with no line of shader in it |
| `swarm.frost` | Five hundred and twelve things, lit by lamps that stop |
| `spin.frost` | A plugin that turns things, which is a game's idea rather than an engine's |

Frost has no crates, so the four names and the order they may reach in are
declared in `frost.json`, and both compilers refuse a crossing while they
resolve the import: `layer: 'lib/renderer' may not reach 'lib/engine'`. The
tests `both_compilers_refuse_a_layer_reaching_upward` and
`both_compilers_refuse_a_layer_reached_through_an_absolute_root` hold the two to
the same answer. A crossing is visible where it is written, because reaching
another layer is a path and reaching a neighbour is a bare name:

```frost,sketch
import "wgpu.frost"              // beside it, in the same layer
import "../platform/sdl.frost"   // a layer below, and it says so
```

```bash
just app window
just app input
just app triangle
just app scene
just app spinning
just app textured
just app shadowed
just app gltf_model
just app lit
just app swarm
just app spin
```

## The SDL binding

`window.frost` is fifty lines and contains no `unsafe` block. It calls
`sdl_init`, `window_create`, `poll_event`, `sdl_delay`, `window_destroy` and
`sdl_quit`, and every one of them is an ordinary Frost function.

An unsafe block is a perimeter. The `extern` declarations, the pointer casts and
the NUL-terminated copies all live in `sdl.frost`, and calling a function that
contains one does not require one. The binding writes twelve of them and the
program writes none.

Beyond declaring the externs, the binding does four things.

A window is `distinct ^u8`, so it cannot be handed to a call that wants some
other pointer, and no pointer from elsewhere can be handed to a call that wants
a window. `is_no_window` asks whether SDL failed, so the only windows a program
holds are the ones SDL made.

A title is a `str`. `window_create` makes the NUL-terminated copy, calls
`SDL_CreateWindow`, and frees the copy, so no caller promises a terminator.

An event kind is an enum. SDL numbers its events with gaps and a program handles
a few of them, so `event_kind` maps the ones the examples need and answers
`Other` for the rest. `SDL_Event` is a 128-byte union with the kind in its first
four bytes. Frost has no unions: the bytes are the value, and the kind is read
out of them.

Flags are `flags` types, because SDL's flags are bits and a call takes several
at once:

```frost,sketch
window_create("Frost", 960, 540,
    WindowFlags::Resizable | WindowFlags::HighPixelDensity)
```

The combination is a `WindowFlags` because both operands are, so no annotation
is written. `InitFlags` and `WindowFlags` are separate types, so a video flag
cannot be passed where a window flag belongs.

`sdl.frost` covers what the graphics examples need, which is a small part of
SDL. A call added to it is one line beside its extern.

## The generated wgpu binding

`lib/renderer/wgpu.frost` is generated. Rebuild it whenever the schema under it
moves:

```bash
just bindgen
just app triangle
```

`just bindgen` builds `tools/wgpu_bindgen.frost` and runs it. It reads
`lib/renderer/wgpu/webgpu.json`, the same file upstream generates
`webgpu.h` from, so the binding follows the schema upstream publishes. It writes
`lib/renderer/wgpu.frost`. Both paths are constants at
the top of the tool, so neither is configurable.

`lib/renderer/wgpu/` is gitignored except for the two the tree carries:
`webgpu.json`, which the bindgen reads, and `wgpu_native.dll`, which a Windows
build links against. `just deps` fetches the rest of a wgpu-native
distribution. On Windows the recipe links `lib/renderer/wgpu/wgpu_native.dll`
and copies it beside the executable. On Unix it links `-lwgpu_native` from the
system. SDL3 is the same story, from the system on Unix and from
`lib/platform/SDL3.dll` on Windows, with `SDL3_DIR` to point the recipe
somewhere else.

`just bindgen` prints ten numbers when it finishes: how many handles,
enumerations, flag families, structs, methods, callback infos, free functions,
constants and initialisers it emitted, and the size of the file.

### What it generates

The tool is written in Frost, over `std/json.frost`, `std/fs.frost`,
`std/format.frost` and `std/strings.frost`. It writes one module with everything
exported, in the shapes Frost prefers:

An object becomes `distinct ^u8` plus a `no_<name>()` for the handle that names
nothing, so an adapter cannot be written where a device belongs. A distinct type
cannot be built from a raw pointer, so `no_adapter()` is the only way to write
one.

A bitflag family becomes one `flags u64` declaration, so a bit is named under
its type and a buffer usage cannot be handed where a texture usage belongs. An
enumeration becomes one screaming-case constant per entry, named for its family
and its entry: `PRESENT_MODE_FIFO`, `LOAD_OP_CLEAR`, `FEATURE_LEVEL_CORE`.

A struct becomes a struct, with an extensible one carrying `next_in_chain: ^u8`
in front and an extension one carrying `chain: Chained`, so an extension is
assignable to the `next_in_chain` of the thing it extends. A C array
member becomes two fields, a `_count` and a pointer, named the way the header
names them. Every struct also gets a `<name>_init()` returning a zeroed one, so
a program names only the fields it means:

```frost,sketch
var configuration := surface_configuration_init()
configuration.device = device
configuration.format = surface_format
configuration.usage = TextureUsage::RenderAttachment
configuration.width = WIDTH
configuration.height = HEIGHT
configuration.present_mode = PRESENT_MODE_FIFO
surface_configure(surface, ptr_to(configuration))
```

A callback becomes a `<name>CallbackInfo` struct holding the function beside its
two userdata slots. The function field is a function type, so a Frost function
is assigned to it directly.

A method becomes two declarations. The `extern` keeps the exact C symbol the
linker looks for and stays unexported, so the wrapper is the one place in a
program that calls C, which is unchecked. The wrapper takes the handle first and
is named for the object and the method as the schema spells them, so
`wgpuDeviceCreateBuffer` is `device_create_buffer(device, descriptor)`.

A wrapper whose method answers a handle checks it. C answers with nothing when
the call fails, and a handle of nothing is no handle, so the generated wrapper
prints `wgpu: device_create_buffer answered with nothing` and dies at that call.

## The triangle

`triangle.frost` is the same source on every platform. It opens an SDL window,
pulls the platform handle out of it (`PROP_WIN32_HWND` on Windows, and the X11
and Cocoa property names are in the binding for the others), builds a wgpu
surface on that handle, and runs a render pass per frame.

It writes no `unsafe` block at all. A descriptor goes to C as `ptr_to` on a
struct the program owns, which is the surface address-of and is checked like any
other place, and a callback writes its context through a `mut` parameter rather
than through a userdata pointer it casts back. Every wgpu and SDL entry point it
uses is a safe Frost function, because the two binding files are the perimeter.

Three details in it carry beyond graphics.

The adapter and device callbacks take `value message: StringView`, the struct
itself. Windows hands a sixteen-byte struct to a callee as a pointer to a copy,
and System V passes it in two registers, so a parameter declared `^u8` reads
`userdata1` out of the wrong register on System V. `value` is the parameter mode
that says "as C passes a struct". See section 12.1 of
[ffi.md](../reference/ffi.md).

The projection is `mat4_perspective_infinite_reverse_z`, not `mat4_perspective`.
WebGPU's clip space runs z from 0 to 1, and a matrix built for the `[-1, 1]`
range puts half the scene behind the near plane without any error at all. One
triangle needs no depth buffer to sort it, so nothing here reads the result; the
projection is the reverse-Z one so that a program starting from this file starts
from the convention the rest of the tree keeps. Both matrices and the three
vertex positions come from `std/math.frost`, so the GPU is handed exactly what
Frost computed.

The window's own event loop is a poll, which is SDL3's API. A C callback
declared through an extern's parameter list takes its context first, and Win32's
`WNDPROC` has no context parameter, so it cannot be declared that way.

## The render graph

Everything past the triangle draws through `graph.frost`. A pass is a name, a
function and the state it records with, and it declares three things: the colour
target it writes, the depth target it writes, and the resources it reads.

```frost,sketch
var g := graph_new(device, 3, 2)
screen := graph_backbuffer(g)
map := graph_depth(g, "shadow map", 1024, 1024, DEPTH_FORMAT)
depth := graph_depth(g, "depth", WINDOW_SIZED, WINDOW_SIZED, DEPTH_FORMAT)

shadow := graph_pass(g, "shadow", draw_things, ptr_to(shadow_scene))
graph_writes_depth(g, shadow, map)

scene := graph_pass(g, "scene", draw_things, ptr_to(lit_scene))
graph_writes_color(g, scene, screen, background)
graph_writes_depth(g, scene, depth)
graph_reads(g, scene, map)

graph_schedule(g)
```

From those declarations `graph_schedule` works out the order: an edge from a
pass that writes a resource to every pass that reads it, and an edge in
declaration order between two passes writing the same one. Kahn's algorithm over
those edges, re-seeded in declaration order every round, so a graph with no
dependencies runs exactly as it was written and two passes over one keep the
order they were declared in.

It answers false, having said which pass and which resource, for a read of a
resource nothing writes, a cycle, a pass that writes nothing at all, and a read
of a resource that follows the window. The last of those is the least obvious: a
bind group naming a texture is made once, and a window-sized texture is thrown
away and remade on a resize, so sampling one leaves the binding pointing at a
texture that no longer exists.

`graph_run` then makes each pass's `Target` and hands its function a
`RenderPassEncoder` over it. The load ops come from the same declarations: every
resource starts each frame unwritten, the first pass to write one clears it, and
every pass after loads what is there. A pass body picks no load op of its own.

### What a resource can be

A colour or depth target, the window's own texture, and two more:

A **buffer** is a run of bytes a pass writes and another reads. It orders the
passes around it, which is all a compute step needs from a graph. It is never an
attachment and decides no load op. `graph_compute_pass` declares one the same
way a drawing pass declares a target, and is handed a compute pass encoder.

`shadowed.frost` uses one: a compute pass writes a value per thing that is
drawn, from the frame's clock and the index, and the scene pass reads the run to
add a glow the bloom then picks up. The compute pass is declared *after* the
pass that reads it and runs first, because the buffer between them is the only
thing that says so.

A **transient** lives inside one frame. The graph works out when each resource
is first and last touched, and hands two transients the same texture when the
first is finished with before the second is started. A chain of post-process
steps costs two textures however long it gets, because each step reads the one
before and writes the next:

```frost,sketch
first  := graph_transient_color(g, "first", 64, 64, format)
middle := graph_transient_color(g, "middle", 64, 64, format)
last   := graph_transient_color(g, "last", 64, 64, format)
// first is free by the time last starts, so they are one texture
```

An **external** is one the graph orders passes around and does not own. A
program hands it a texture with `graph_set_external`, which is how the same
graph draws each camera into that camera's target.

A declared size may be a share of the window: `-1` is the whole of it, `-2` a
half, `-4` a quarter. A chain that runs at half resolution says so once and
follows a resize.

### Ordering a program asks for

`graph_runs_after(g, after, before)` puts one pass after another when they share
no resource and the order still matters. It joins the edges the reads and writes
imply, and a cycle made this way is refused like any other.

### Turning passes off, and phases

`graph_set_enabled` takes a pass out of the frame without changing the order, so
a debug view toggles one every frame with no rescheduling. `graph_save_enabled`
and `graph_restore_enabled` put back the state each pass had, so a pass that was
already off stays off.

A pass belongs to a phase, and `graph_run_phase` runs one. Rendering the same
graph once per camera goes through phases: the shared work carries one phase,
each view's work carries its own, and a program runs the shared phase once and
each view's phase with a different external bound.

### Sub-graphs

A graph built on its own can be handed to another with `graph_sub`, and
`graph_sub_pass` is a pass that runs it. `shadowed.frost` uses one for a second
camera: a child graph with its own target, its own depth and its own scene pass,
drawn into a corner of the window by a pass that reads what the child wrote.

A sub-graph cannot be reached once it has been handed over, so `graph_resize`
reaches its children. A child whose resources are a fixed size makes them on the
first of those and ignores the rest; one whose resources follow the window
follows the same window its parent does. The pass declares what it reads and
writes like any other, so the nested work lands in the right place in the
parent's order, and `graph_takes` binds one of the child's resources to one of
the parent's.

A `Graph` is `linear` for the same reason a `Frame` is: `graph_sub` keeps the
child, so a caller that also freed its own copy would free the same tables
twice, and the obligation on the type refuses that at compile time.

Ordering, resource lifetimes and pool assignment are all arithmetic over tables
and touch no device, so `graph.frost` carries twenty-four `test` blocks that run
under `just test` with `no_device()` in place of a GPU: what runs first, which
transients share a texture, what the load ops come out as, the phase and enabled
state a pass carries, and each of the five graphs that cannot run at all.

## From a world to a run

Every demo past the triangle draws what an ECS world holds, and the walk from
one to the other is where two of the layers meet.

`lib/renderer/render_world.frost` is the renderer's side. It asks an entity for
two components and no more:

```frost
import "math.frost"
Transform :: struct { matrix: Mat4 }
Drawn :: struct { mesh: i64, material: i64, layer: i64 }
```

Where a thing ended up is somebody else's answer by the time it arrives, so the
renderer knows nothing of placements, trees, clocks and windows. Beside those
two sits the `RenderWorld`: the object and transform runs a frame fills, and
the GPU buffers they are uploaded to.

`lib/engine/world.frost` is the other side. A thing there has a placement, a
turn and somewhere it hangs from, and a frame works out where it ended up.
`lib/engine/scene_sync.frost` fills the run from that. Both reach down for the
two components above, and nothing on the renderer's side reaches back.

`world_prepare` registers both of those along with the two the far side owns:

```frost
import "math.frost"
LocalTransform :: struct { translation: Vec3, rotation: Quat, scale: Vec3 }
GlobalTransform :: struct { matrix: Mat4 }
```

`world_schedule` is the frame. `move_camera` and `turn_things` run in `First`,
and `place_things` walks down from every root in `Update`, leaving each thing's
`GlobalTransform` with whatever it hangs off applied over its own
`LocalTransform`, however deep it hangs. A system is a `fn(mut World)` and captures nothing, so which component
is which travels in a `WorldIds` resource.

### What a system knows about the machine

`platform.frost` splits what a frame saw from what owns the window. `Platform`
holds the window, the event queue and the clock. `Input` is the state a poll
left behind, and `platform_input` hands out a copy. A loop puts that copy in the
world:

```frost,sketch
world_input(world, platform_input(p))
schedule_run(frame_schedule, world, ANY_STATE)
write_frame(queue, frame_uniform, world_camera(world), width(p), height(p))
```

`move_camera` reads the `Input` and the `Camera` out of the world, moves one by
the other, and writes it back. It captures nothing and holds no window. How long
the frame took rides along in the `Input`, so the clock has one reading and
everything paced by it reads the same one: `turn_things` takes its step from
there too.

Every accessor comes in two forms, `key_down(p, KEY_W)` and
`input_key_down(held, KEY_W)`, because a main loop has the window in hand and a
system has only what it was given. The first is a one-line call to the second.

The camera lives in the world for the same reason: `world_camera_set` puts it
where a program wants it to start, the schedule moves it, and the loop reads it
back to write the frame uniform.

Then `scene_sync` walks the world once and leaves a `DrawList` behind:

```frost,sketch
scene_sync(world, render, registry, cache, ids)
```

Nothing per entity is made or kept. The walk fills one flat run, one entry per
thing, built from nothing every frame: a thing spawned while the program runs
is drawn on the next frame because the walk finds it, and a thing despawned
stops being drawn because the walk does not. There is no table to resize and
nothing to renumber.

A pass binds the run once and draws a class at a time:

```frost,sketch
render_pass_encoder_set_bind_group(pass, 0, s.frame_group, 0, no_pointer())
render_pass_encoder_set_bind_group(pass, 1, render_world_group(held), 0,
    no_pointer())
render_pass_encoder_set_bind_group(pass, 2, s.light_group, 0, no_pointer())

render_pass_encoder_set_pipeline(pass, s.opaque_pipeline)
render_world_draw_class(pass, held, CLASS_OPAQUE)
```

The run lives in a buffer the shader indexes, so a thing's transform and its
material reach the GPU without a bind group of its own. A class is a pipeline's
worth of the run, and the six of them are the order a frame draws in.

### Layers, and one run per pass

A thing carries the layer it belongs to, and `drawn_in_layer` builds a run from
one of them. `scene.frost` divides its world that way: two calls over one slot
table give the near pass and the far pass a run apiece, each holding only what
that pass draws. Which things those are is decided once a frame in the walk, and
a pass records every item it is handed.

### A model out of a file

`gltf.frost` reads a binary glTF into geometry, materials and a tree of nodes,
and `gltf_spawn` turns that into entities:

```frost,sketch
var held := gltf_read("lib/engine/assets/shapes.glb")
root := gltf_spawn(world, held, device, queue, registry, cache, ids)
gltf_free(held)
```

One entity per node, hung off whatever the file hung it off, and one child per
primitive under a node that draws, because a glTF mesh is a list of primitives
and each carries its own material. The geometry and the materials go to the
device once, here. The entities carry the numbers they came back as, so after
this the file has done its job and nothing spawned points into it.

A node that draws nothing has no `Drawn` component at all. A query asks for the
components a thing carries, so giving such a node that component would put it in
front of a pass with a mesh that is not there. Most of a real file's nodes draw
nothing: they are the joints and the groupings the model was built out of.

`Placement` carries a rotation for this. A glTF node gives a quaternion, and
`Spin` is a turn applied on top of it about an axis of the thing's own, so a
loaded model that already faces somewhere can also be made to rotate with
neither turn folded into the other. `just app gltf_model` puts a spin on the one
entity the whole file hangs off, which is how a model turns as one thing while
its own tree stays untouched.

### Grouping by material

`Drawn` carries a material number, and the run is sorted by it so the things
sharing one sit together. `textured.frost` puts its four images in one
`texture_2d_array`, binds group 2 once for the whole frame, and gives each of
its thirty-six things the layer number its material names, so the count of draws
is the count of meshes. A material means whatever its pass decides: `textured`
reads an image layer for it, and `spinning` reads the colour already in the
uniform.

## The frame is linear

Every WebGPU object is reference counted: `wgpuTextureRelease` and its
twenty-two siblings exist for exactly that, and a surface cannot be configured
while a texture of its swapchain is still held. A renderer that acquires a
surface texture each frame and never gives it back draws correctly and then dies
on the first window resize with `Invalid surface`.

Rust reaches for `Drop` here. Frost has no destructors, only the obligation on
the type:

```frost,sketch
Frame :: linear struct {
    texture: SurfaceTexture,
    view: TextureView,
    encoder: CommandEncoder,
    ready: bool,
}

renderer_end :: fn(mut r: Renderer, move f: Frame) { ... }
```

`renderer_begin` hands back a linear value and `renderer_end` takes it by
`move`, so the compiler checks the pairing. This shape does not compile:

```frost,sketch
var f := renderer_begin(r)
if (frame_ok(f)) {
    graph_run(graph, f)
    renderer_end(r, f)      // ends the frame only when one was acquired
}
```

> linearity: linear value 'f' is not consumed on every path before return

The fix is to end the frame outside the `if`. Nothing is inserted on your
behalf, and the obligation falls only on the types written `linear`, which here
are the frame-scoped handles. `Device`, `BindGroup` and the pipelines are
refcounted and read out of structs all over a program, which is shared
ownership, and marking those linear would model the wrong thing.
