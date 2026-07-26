# The math library

`std/math.frost` is a small single-precision math library for graphics and
games: vectors, matrices, and quaternions as plain structs with free functions
over them. It is an ordinary Frost library, not a language feature. It needs no
compiler support, imports with `import "math.frost"`, and compiles and runs on
both backends of both compilers.

It is written the way the rest of the language is. There is no operator
overloading and there are no methods, so it is `vec3_add(a, b)` rather than
`a + b` and `mat4_mul(a, b)` rather than `a * b`. The type prefix on each name
(`vec3_`, `mat4_`, `quat_`) is the namespace, the same convention the rest of the
standard library follows. Every value is `f32`.

`std/math64.frost` is the same library at double precision: the same shapes, the
same rules, the same column-major convention, with `f64` in place of `f32` and a
`d` on every name (`Vec3d`, `vec3d_add`, `mat4d_perspective`, `radiansd`). The
two can be imported together, since no name is shared.

The transcendentals it needs (`sqrtf`, `sinf`, `cosf`, `tanf`) are the C standard
library's single-precision ones, declared `safe extern` because each takes and
returns a number with no pointer. On a POSIX host they live in `libm`. On Windows
they are in the C runtime and need no extra link flag.

## Types

```frost
Vec2 :: struct { x: f32, y: f32 }
Vec3 :: struct { x: f32, y: f32, z: f32 }
Vec4 :: struct { x: f32, y: f32, z: f32, w: f32 }
Mat4 :: struct { m: [16]f32 }
Quat :: struct { x: f32, y: f32, z: f32, w: f32 }
```

They are plain data, passed and returned by value like any other struct. A `Mat4`
is its sixteen floats in a single array field, and a `Quat` stores its vector
part in `x, y, z` with the scalar part in `w`.

## Vectors

`vec2`, `vec3`, and `vec4` construct a vector from its components. The operations
are what a geometry pass reaches for:

- `vec3_add`, `vec3_sub`, `vec3_scale`, `vec3_neg`: componentwise arithmetic and
  scaling by a float. `vec2` and `vec4` carry the add/sub/scale set too.
- `vec3_dot`, `vec3_cross`: the dot product (a float) and the cross product (the
  perpendicular vector). `vec2_dot` and `vec4_dot` exist for their dimensions.
  The cross product is `Vec3` only.
- `vec3_length`, `vec3_length_sq`, `vec3_distance`: the length, its square (which
  skips the square root when only a comparison is wanted), and the distance
  between two points.
- `vec3_normalize`: the unit vector in the same direction, and the zero vector if
  the input has no length rather than a division by zero.
- `vec3_lerp`: the linear interpolation from `a` to `b` by `t`.

## Matrices

`Mat4` is column-major, the OpenGL and Vulkan convention. The element at row `r`,
column `c` lives at `m.m[c * 4 + r]`. `mat4_mul(a, b)` is the product that applies
`b` then `a` to a column vector, so a model-view-projection is
`mat4_mul(projection, mat4_mul(view, model))`, read right to left the way the
transforms apply.

- `mat4_zero`, `mat4_identity`: the constant matrices.
- `mat4_mul`, `mat4_mul_vec4`: matrix product and matrix-times-vector.
- `mat4_translation`, `mat4_scale`, `mat4_rotation_x`, `mat4_rotation_y`,
  `mat4_rotation_z`: the standard affine builders, each from a `Vec3` or an angle
  in radians.
- `mat4_transform_point`, `mat4_transform_dir`: carry a `Vec3` through a matrix. A
  point carries an implicit `w` of 1, so a translation moves it. A direction
  carries `w` of 0, so a translation leaves it alone.
- `mat4_perspective`, `mat4_perspective_zo`, `mat4_ortho`, `mat4_look_at`: a
  right-handed perspective projection into the `[-1, 1]` depth range (the OpenGL
  clip convention), the same projection into `[0, 1]` (what Direct3D, Metal,
  Vulkan and WebGPU take, and what `examples/graphics/triangle.frost` uses), an
  orthographic projection, and a view matrix looking from an eye toward a center.

  The depth range is the one thing here that is not a matter of taste. A matrix
  built for the wrong one puts half the scene behind the near plane, and it does
  so without any error, so the projection has to match the API being drawn with.

## Quaternions

A `Quat` represents a rotation and composes without the gimbal lock of Euler
angles.

- `quat_identity`, `quat_from_axis_angle`: the no-rotation quaternion and a
  rotation of an angle about a (normalized) axis.
- `quat_mul`: compose two rotations.
- `quat_length`, `quat_normalize`, `quat_conjugate`: the norm, the unit
  quaternion, and the conjugate (which inverts a unit rotation).
- `quat_rotate_vec3`: rotate a vector by a quaternion, computed as
  `q * (v, 0) * conjugate(q)`.
- `quat_to_mat4`: the equivalent rotation matrix, for handing a quaternion to code
  that wants a `Mat4`.

## Angles

`radians` and `degrees` convert between the two. Angles read more naturally in
degrees, but the transcendentals want radians, so a rotation is
`mat4_rotation_y(radians(90.0))`.

## In action

`examples/native/math_transform.frost` builds a camera and a model transform out
of these pieces and pushes a point through a full model-view-projection, printing
results as scaled integers so the Cranelift and C backends agree on the output
byte for byte. The test `self_hosted_standard_library_math` compiles the library
through the self-hosted compiler on both backends.

```frost
import "io.frost"
import "math.frost"

main :: fn() -> i64 {
    // A quaternion of ninety degrees about Y turns +Z into +X.
    turn := quat_from_axis_angle(vec3(0.0, 1.0, 0.0), radians(90.0))
    spun := quat_rotate_vec3(turn, vec3(0.0, 0.0, 1.0))
    scaled : i64 = spun.x * 1000.0   // ~1000
    print_int_line(scaled)
    0
}
```

## Tests

Every exported function has a `test` block beside it in `std/math.frost`, run
with:

```bash
frost --test std/math.frost
```

The same twenty run over `std/math64.frost`, which is where a copy that changed
a formula shows up. The suite runs both through both backends of both
compilers. A differential test
would only say the backends agree, and a rotation that turns the wrong way, a
projection with its depth range inverted and a quaternion that is its own
inverse all agree across backends while all being wrong, so these check the
answers rather than the agreement.

Results are compared within a tolerance, because a square root or a
trigonometric call does not land on an exact float. Where a rotation can be
expressed two ways the tests check the two against each other, so
`quat_to_mat4` has to be the rotation `quat_rotate_vec3` applies and not its
inverse.

## Which precision

`std/math.frost` is what a renderer wants. Vertices, per-frame transforms and
anything crossing to a GPU are single-precision, and it is half the bytes.

`std/math64.frost` is for the places where range or accumulated error decides
the answer: a simulation stepping for hours, world coordinates far from the
origin, a solver. Reach for it there and convert at the boundary.

It is a second copy rather than one library generic over the element type. A
generic one would have to take the transcendentals as compile-time arguments at
every call, since `sqrt` and `sqrtf` are different C functions, which costs more
at every use than a copy costs once. That is the trade
[philosophy.md](philosophy.md) names: no traits, so write the one you need over
the layout you have. The copy is mechanical and the same twenty tests run over
both, so a formula that survived the copy wrong fails.

## What is not here

The library is value-typed on purpose. It does not provide SIMD-packed vectors,
and the language is not going to grow them: the layout that makes vectorization
possible is what `columns<T, N>` and the `inline` marker already give a C
compiler, and [roadmap.md](roadmap.md) says why intrinsics would put the work
where it pays least. Nor does it provide a general N-dimensional matrix. It is the graphics math a renderer and a game loop
actually reach for, and nothing more.
