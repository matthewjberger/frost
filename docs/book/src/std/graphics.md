# Graphics bindings

`examples/graphics/` is not part of `std/`. It is two C libraries bound to
Frost, and two programs that use them: a window, and a spinning triangle drawn
with wgpu. It is here because those bindings are the working answer to what a
Frost program does when it has to talk to a real graphics API, and because the
math the triangle uses comes straight out of [math.md](math.md).

| File | What it is |
| --- | --- |
| `sdl.frost` | SDL3: a window, its events, and the clock that paces them. Hand-written |
| `wgpu.frost` | The whole WebGPU API. Generated, and not committed |
| `window.frost` | Opens a window and pumps it until it is closed |
| `triangle.frost` | Draws a rotating triangle into that window |

```bash
just window
just triangle
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
