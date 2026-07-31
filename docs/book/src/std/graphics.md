# Graphics bindings

`examples/graphics/` sits outside `std/`. It is two C libraries bound to Frost,
the pieces a drawing program is built out of, and six programs that use them. It
is here because those bindings are the working answer to what a Frost program
does when it has to talk to a real graphics API, and because the math they use
comes straight out of [math.md](math.md).

| File | What it is |
| --- | --- |
| `sdl.frost` | SDL3: a window, its events, and the clock that paces them. Hand-written |
| `wgpu.frost` | The whole WebGPU API. Generated, and not committed |
| `platform.frost` | The window, the keyboard and the frame clock under one handle |
| `renderer.frost` | The device, the surface and the frame under one handle |
| `graph.frost` | A render graph: passes declare their targets, the order follows |
| `camera.frost` | Where the eye is and what it looks at |
| `mesh.frost` | Geometry on the device, and the cache a program names it through |
| `material.frost` | A registry of surfaces, so a thing that is drawn carries a number |
| `texture.frost` | Images on the device, and render targets to draw into |
| `window.frost` | Opens a window and pumps it until it is closed |
| `triangle.frost` | Draws a rotating triangle into that window |
| `scene.frost` | Entities in an ECS, two passes, depth deciding what is in front |
| `spinning.frost` | Lit surfaces: a mesh cache, a material registry, two bind groups |
| `textured.frost` | The same field with its surfaces read off an image |
| `shadowed.frost` | A shadow pass and a scene pass, ordered by the map between them |

```bash
just window
just triangle
just scene
just spinning
just textured
just shadowed
```

## The binding is the perimeter

`window.frost` is fifty lines and contains no `unsafe` block. It calls
`sdl_init`, `window_create`, `poll_event`, `sdl_delay`, `window_destroy` and
`sdl_quit`, and every one of them is an ordinary Frost function.

That is the point of a binding file. An unsafe block is a perimeter: the
`extern` declarations, the pointer casts and the NUL-terminated copies all live
in `sdl.frost`, and calling a function that contains one does not require one.
Ten unsafe blocks in the binding buy zero in the program.

Beyond declaring the externs, the binding does four things.

A window is `distinct ^u8` rather than a bare pointer, so it cannot be handed to
a call that wants some other pointer, and no pointer from elsewhere can be
handed to one that wants a window. `is_no_window` asks whether SDL failed rather
than handing out a null window, so the only windows a program holds are the ones
SDL made.

A title is a `str`. `window_create` makes the NUL-terminated copy, calls
`SDL_CreateWindow`, and frees the copy, so no caller promises a terminator.

An event kind is an enum. SDL numbers its events with gaps and a program handles
a few of them, so `event_kind` maps the ones the examples need and answers
`Other` for the rest, which means a kind a program did not ask about is still a
kind it can see. `SDL_Event` is a 128-byte union with the kind in its first four
bytes; Frost has no unions and does not need one, because the bytes are the
value and the kind is read out of them.

Flags are `flags` types rather than enums, because SDL's flags are bits and a
call takes several at once:

```frost
window_create("Frost", 960, 540,
    WindowFlags::Resizable | WindowFlags::HighPixelDensity)
```

The combination is a `WindowFlags` because both operands are, so nothing has to
say so. `InitFlags` and `WindowFlags` are separate types, so a video flag cannot
be passed where a window flag belongs.

What is not covered is most of SDL. `sdl.frost` is what the two graphics
examples need, and a call added to it is one line beside its extern.

## The wgpu binding is generated, and not in the repository

`examples/graphics/wgpu.frost` is gitignored. It has to be built before
`just triangle` will compile:

```bash
just bindgen
just triangle
```

`just bindgen` builds `tools/wgpu_bindgen.frost` and runs it. It reads
`examples/graphics/wgpu/webgpu.json`, the same file upstream generates
`webgpu.h` from, so the binding follows the schema rather than a parse of the C
header. It writes `examples/graphics/wgpu.frost`. Both paths are constants at
the top of the tool, so neither is configurable.

`examples/graphics/wgpu/` is gitignored too, all of it. A reader has to put a
wgpu-native distribution there: `webgpu.json` for the bindgen to read, and the
library itself to link against. On Windows the `triangle` recipe links
`examples/graphics/wgpu/wgpu_native.dll` and copies it beside the executable. On
Unix it links `-lwgpu_native` from the system. SDL3 is the same story, from the
system on Unix and from `examples/graphics/SDL3.dll` on Windows, with `SDL3_DIR`
to point the recipe somewhere else.

`just bindgen` prints ten numbers when it finishes: how many handles,
enumerations, flag families, structs, methods, callback infos, free functions,
constants and initialisers it emitted, and the size of the file.

### What it generates

The tool is written in Frost, over `std/json.frost`, `std/fs.frost`,
`std/format.frost` and `std/strings.frost`. What comes out is one module with
everything exported, in the shapes Frost prefers rather than the shapes C spells
them in:

An object becomes `distinct ^u8` plus a `no_<name>()` for the handle that names
nothing, so an adapter cannot be written where a device belongs. A distinct type
cannot be built from a raw pointer, so `no_adapter()` is the only way to write
one, and the checks around it are what keep it from ever standing for a real
handle.

A bitflag family becomes one `flags u64` declaration, so a bit is named under
its type and a buffer usage cannot be handed where a texture usage belongs. An
enumeration becomes one screaming-case constant per entry, named for its family
and its entry: `PRESENT_MODE_FIFO`, `LOAD_OP_CLEAR`, `FEATURE_LEVEL_CORE`.

A struct becomes a struct, with an extensible one carrying `next_in_chain: ^u8`
in front and an extension one carrying `chain: Chained`, which is what makes an
extension assignable to the `next_in_chain` of the thing it extends. A C array
member becomes two fields, a `_count` and a pointer, named the way the header
names them. Every struct also gets a `<name>_init()` returning a zeroed one, so
a program names only the fields it means:

```frost
mut configuration := surface_configuration_init()
configuration.device = device
configuration.format = surface_format
configuration.usage = TextureUsage::RenderAttachment
configuration.width = WIDTH
configuration.height = HEIGHT
configuration.present_mode = PRESENT_MODE_FIFO
surface_configure(surface, ptr_to(configuration))
```

A callback becomes a `<name>CallbackInfo` struct holding the function beside its
two userdata slots. The function field is a function type rather than a pointer,
so a Frost function is assigned to it directly.

A method becomes two declarations. The `extern` keeps the exact C symbol,
because that is what the linker will look for, and it is not exported: calling C
is unchecked, and the wrapper is meant to be the one place in a program that
does it. The wrapper takes the handle first and is named for the object and the
method as the schema spells them, so `wgpuDeviceCreateBuffer` is
`device_create_buffer(device, descriptor)`.

A wrapper whose method answers a handle checks it. C answers with nothing when
it fails, and a handle of nothing is not a handle, so the generated wrapper
prints `wgpu: device_create_buffer answered with nothing` and dies there rather
than letting every later call read through it. The failure is said where it
happened rather than at whichever call happened to be next.

## The triangle

`triangle.frost` is the same source on every platform. It opens an SDL window,
pulls the platform handle out of it (`PROP_WIN32_HWND` on Windows, and the X11
and Cocoa property names are in the binding for the others), builds a wgpu
surface on that handle, and runs a render pass per frame.

It writes `unsafe` twenty-one times, all of them either `ptr_cast` to hand a
descriptor's address to C as a `^u8`, or a write through the userdata pointer
inside a callback. None of it is a call into wgpu: every wgpu and SDL entry
point it uses is a safe Frost function, because the two binding files are the
perimeter. The remaining blocks are what it costs to pass a struct C wants a
pointer to.

Three details in it are worth reading for reasons beyond graphics.

The adapter and device callbacks take `value message: StringView`, the struct
itself rather than a pointer to it. Windows hands a sixteen-byte struct to a
callee as a pointer to a copy, so declaring the parameter `^u8` worked there and
would have read `userdata1` out of the wrong register on System V, where the
struct takes two. `value` is the parameter mode that says "as C passes a
struct". See section 12.1 of [ffi.md](../reference/ffi.md).

The projection is `mat4_perspective_zo`, not `mat4_perspective`. WebGPU's clip
space runs z from 0 to 1, and the wrong one puts half the scene behind the near
plane without any error at all. Both matrices and the three vertex positions
come from `std/math.frost`, so what the GPU is handed is what Frost computed
rather than a table of constants written into the shader.

The window's own event loop is a poll rather than a callback, which is SDL3's
API. A C callback declared through an extern's parameter list must take its
context first, so Win32's `WNDPROC`, which has no context parameter, could not
be declared that way.

## The render graph

Everything past the triangle draws through `graph.frost`. A pass is a name, a
function and the state it records with, and it declares three things: the colour
target it writes, the depth target it writes, and the resources it reads.

```frost
mut g := graph_new(device, 3, 2)
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
resource nothing writes, a cycle, a pass that writes nothing, and a read of a
resource that follows the window. That last one is the least obvious and the
most useful: a bind group naming a texture is made once, and a window-sized
texture is thrown away and remade on a resize, so sampling one leaves the
binding pointing at a texture that no longer exists.

`graph_run` then makes each pass's `Target` and hands its function a
`RenderPassEncoder` over it. The load ops come from the same declarations: every
resource starts each frame unwritten, the first pass to write one clears it, and
every pass after loads what is there. In `scene.frost` that rule replaced a pass
body reading its own state to pick between two load ops, which is the kind of
rule a reader keeps rather than one the code does.

Scheduling is arithmetic over two tables and touches no device, so
`graph.frost` carries eight `test` blocks that run under `just test` with
`no_device()` in place of a GPU: what runs first, what the load ops come out as,
and each of the four graphs that cannot run at all.

## The frame is linear

Every WebGPU object is reference counted: `wgpuTextureRelease` and its
twenty-two siblings exist for exactly that, and a surface cannot be configured
while a texture of its swapchain is still held. A renderer that acquires a
surface texture each frame and never gives it back draws correctly and then dies
on the first window resize with `Invalid surface`.

Rust reaches for `Drop` here. Frost has no destructors, and what it has instead
is the obligation on the type:

```frost
Frame :: linear struct {
    texture: SurfaceTexture,
    view: TextureView,
    encoder: CommandEncoder,
    ready: bool,
}

renderer_end :: fn(mut r: Renderer, move f: Frame) { ... }
```

`renderer_begin` hands back a linear value and `renderer_end` takes it by
`move`, so the pairing is checked rather than remembered. Marking the type was
enough to make all four demos stop compiling, each on the same shape:

```frost
mut f := renderer_begin(r)
if (frame_ok(f)) {
    graph_run(graph, f)
    renderer_end(r, f)      // ends the frame only when one was acquired
}
```

> linearity: linear value 'f' is not consumed on every path before return

The fix is to end the frame outside the `if`. This is the trade the design makes
in place of a destructor: nothing is inserted on your behalf, and nothing is
forgotten either. It only pays where the type is marked, and the handles worth
marking are the frame-scoped ones. `Device`, `BindGroup` and the pipelines are
refcounted and read out of structs all over a program, which is shared
ownership; making those linear would model the wrong thing.
