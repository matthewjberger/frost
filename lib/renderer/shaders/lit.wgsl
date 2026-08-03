struct Frame {
    view_projection: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_position: vec4<f32>,
    ambient: vec4<f32>,
    // How many lights there are, and how many of those come first because they
    // are directional.
    counts: vec4<u32>,
}

struct ClusterUniforms {
    inverse_projection: mat4x4<f32>,
    screen_size: vec2<f32>,
    z_near: f32,
    z_far: f32,
    cluster_count: vec4<u32>,
    tile_size: vec2<f32>,
    num_lights: u32,
    num_directional_lights: u32,
}

struct LightGrid {
    offset: u32,
    count: u32,
}

struct Light {
    position: vec4<f32>,
    direction: vec4<f32>,
    color: vec4<f32>,
    light_type: u32,
    range: f32,
    inner_cone: f32,
    outer_cone: f32,
    shadow_index: i32,
    light_size: f32,
    cookie_layer: u32,
    pad_zero: f32,
}

struct ObjectData {
    transform_index: u32,
    mesh_id: u32,
    material_id: u32,
    batch_id: u32,
    pipeline_class: u32,
    visible: u32,
    is_overlay: u32,
    pad_zero: u32,
}

struct MaterialData {
    base_color: vec4<f32>,
    roughness: f32,
    metallic: f32,
    alpha_cutoff: f32,
    flags: u32,
}

const FLAG_MASK: u32 = 2u;
const PI: f32 = 3.14159265358979323846;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(1) @binding(0) var<storage, read> objects: array<ObjectData>;
@group(1) @binding(1) var<storage, read> visible: array<u32>;
@group(1) @binding(2) var<storage, read> transforms: array<mat4x4<f32>>;
@group(1) @binding(3) var<storage, read> materials: array<MaterialData>;
@group(2) @binding(0) var<storage, read> lights: array<Light>;
@group(2) @binding(1) var<storage, read> light_grid: array<LightGrid>;
@group(2) @binding(2) var<storage, read> light_indices: array<u32>;
@group(2) @binding(3) var<uniform> cluster: ClusterUniforms;

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) object_index: u32,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>,
           @location(1) normal: vec3<f32>,
           @builtin(instance_index) instance: u32) -> VertexOut {
    var out: VertexOut;
    let object_index = visible[instance];
    let object = objects[object_index];
    let model = transforms[object.transform_index];
    let world = model * vec4<f32>(position, 1.0);
    out.clip = frame.view_projection * world;
    out.world = world.xyz;
    out.normal = (model * vec4<f32>(normal, 0.0)).xyz;
    out.object_index = object_index;
    return out;
}

// How much of one light reaches a point, and from where.
//
// The two attenuations are nightshade's, and the first is not the obvious one.
// A light with no range does not fall off at all: it is a light a scene wants
// everywhere, and dividing it by the square of the distance would make that
// impossible to ask for. A light with a range falls off with the square of the
// distance inside a window that reaches zero exactly at the range, so it stops
// where it says it stops rather than trailing off forever below the point of
// being visible.
//
// The window is the fourth power of the distance over the range, which is
// glTF's, and it is what keeps the curve close to a true inverse square until
// near the end instead of bending the whole way along.
struct Reaching {
    toward: vec3<f32>,
    strength: f32,
}

fn range_attenuation(range: f32, distance: f32) -> f32 {
    if (range <= 0.0) {
        return 1.0;
    }
    let bounded = max(distance, 0.01);
    let window = max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0);
    return window / (bounded * bounded);
}

// Full strength inside the inner cone, nothing outside the outer one, and a
// smooth step between. Both angles arrive as cosines, so a wider angle is a
// smaller number.
fn spot_attenuation(toward_light: vec3<f32>, facing: vec3<f32>,
                    outer_cone: f32, inner_cone: f32) -> f32 {
    let along = dot(normalize(facing), normalize(-toward_light));
    if (along > outer_cone) {
        if (along < inner_cone) {
            return smoothstep(outer_cone, inner_cone, along);
        }
        return 1.0;
    }
    return 0.0;
}

fn reaching(light: Light, at: vec3<f32>) -> Reaching {
    var out: Reaching;
    let intensity = light.color.w;
    if (light.light_type == 0u) {
        out.toward = normalize(-light.direction.xyz);
        out.strength = intensity;
        return out;
    }
    let offset = light.position.xyz - at;
    let distance = length(offset);
    out.toward = offset / max(distance, 0.0001);

    var fade = range_attenuation(light.range, distance);
    if (light.light_type == 2u) {
        fade = fade * spot_attenuation(offset, light.direction.xyz,
            light.outer_cone, light.inner_cone);
    }
    out.strength = intensity * fade;
    return out;
}

// The prepass. No colour target and no shading: what it produces is depth, and
// where it lands is the whole of what the pass after it reads.
//
// The two multiplications are in the order `vs_main` does them, and that is not
// a matter of taste. Reassociating them is exact in arithmetic and not in
// floating point, so a depth written one way round and tested the other way
// round disagrees in the last bits and the surface speckles where the two
// answers straddle the comparison.
@vertex
fn vs_depth(@location(0) position: vec3<f32>,
            @builtin(instance_index) instance: u32) -> @builtin(position) vec4<f32> {
    let object = objects[visible[instance]];
    let model = transforms[object.transform_index];
    let world = model * vec4<f32>(position, 1.0);
    return frame.view_projection * world;
}

// The microfacet terms. A surface is modelled as a field of mirrors too small
// to see: how many of them face the halfway direction between the light and the
// eye, how much they shadow one another, and how reflective they are at this
// angle. Roughness is what spreads the facets out.
//
// These three are nightshade's `DistributionGGX`, `V_SmithGGXCorrelated` and
// `fresnelSchlick`. The second is a visibility term rather than a geometry one:
// it has the specular denominator already divided out, so the product below is
// three factors rather than four over one.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn visibility_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let toward_eye = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let toward_light = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(toward_eye + toward_light, 0.00001);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(max(1.0 - cos_theta, 0.0), 5.0);
}

// What one light adds, as the metallic-roughness model has it: a specular lobe
// off the facets and a diffuse term from what got through them.
//
// The split between the two is the Fresnel term rather than a number a program
// picks. What is reflected cannot also be scattered, so the diffuse weight is
// what the reflection left; and a metal has no diffuse at all, which is what
// `metallic` scales away. A dielectric reflects about four percent of the light
// head-on whatever it is made of, which is where the 0.04 comes from; a metal
// reflects its own colour, so its base colour is its reflectance.
fn shade(light: Light, at: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
         albedo: vec3<f32>, roughness: f32, metallic: f32) -> vec3<f32> {
    let held = reaching(light, at);
    let l = held.toward;
    let n_dot_l = max(dot(n, l), 0.0);
    if (n_dot_l <= 0.0 || held.strength <= 0.0) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let h = normalize(l + v);
    let n_dot_v = max(dot(n, v), 0.0001);
    let radiance = light.color.rgb * held.strength;

    let f0 = mix(vec3<f32>(0.04, 0.04, 0.04), albedo, metallic);
    let fresnel = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let specular = distribution_ggx(max(dot(n, h), 0.0), roughness)
        * visibility_smith(n_dot_v, n_dot_l, roughness) * fresnel;
    let scattered = (vec3<f32>(1.0, 1.0, 1.0) - fresnel) * (1.0 - metallic);
    let diffuse = scattered * albedo / PI;

    return (diffuse + specular) * radiance * n_dot_l;
}

// Which box of the grid a fragment fell in. The tile is where it is on the
// screen; the slice is how far away it is, on the same logarithmic scale the
// boxes were built with, so the two agree about which box is which.
fn cluster_index(frag: vec2<f32>, view_depth: f32) -> u32 {
    let ratio = log(cluster.z_far / cluster.z_near);
    let bounded = max(view_depth, cluster.z_near);
    let slice = u32(log(bounded / cluster.z_near) / ratio
        * f32(cluster.cluster_count.z));
    let z = clamp(slice, 0u, cluster.cluster_count.z - 1u);
    let x = clamp(u32(frag.x / cluster.tile_size.x), 0u,
        cluster.cluster_count.x - 1u);
    let y = clamp(u32(frag.y / cluster.tile_size.y), 0u,
        cluster.cluster_count.y - 1u);
    return x + y * cluster.cluster_count.x
        + z * cluster.cluster_count.x * cluster.cluster_count.y;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let object = objects[in.object_index];
    let material = materials[object.material_id];
    // A masked surface is kept or dropped whole. It is drawn in the opaque
    // classes and writes depth, so the fragment has to go before anything else
    // does rather than being blended away.
    if ((material.flags & FLAG_MASK) != 0u
        && material.base_color.a < material.alpha_cutoff) {
        discard;
    }
    let facing = normalize(in.normal);
    let toward_eye = normalize(frame.camera_position.xyz - in.world);
    // A roughness of zero is a perfect mirror, and the distribution divides by
    // its square, so the floor is what keeps a highlight a highlight rather
    // than a division by nothing.
    let rough = clamp(material.roughness, 0.045, 1.0);
    let metal = clamp(material.metallic, 0.0, 1.0);

    var lit = material.base_color.rgb * frame.ambient.rgb;

    // A directional light reaches the whole world, so the grid has nothing to
    // say about it and every one of them is walked.
    let directional = frame.counts.y;
    for (var i = 0u; i < directional; i = i + 1u) {
        lit = lit + shade(lights[i], in.world, facing, toward_eye,
            material.base_color.rgb, rough, metal);
    }

    // The rest are found through the box this fragment fell in, which holds the
    // few that reach it out of however many the scene has.
    let view_depth = -(frame.view * vec4<f32>(in.world, 1.0)).z;
    let box = light_grid[cluster_index(in.clip.xy, view_depth)];
    for (var i = 0u; i < box.count; i = i + 1u) {
        let which = light_indices[box.offset + i];
        lit = lit + shade(lights[directional + which], in.world, facing,
            toward_eye, material.base_color.rgb, rough, metal);
    }
    return vec4<f32>(lit, material.base_color.a);
}
