use crate::types::Type;

// How C returns a struct, which is not how Frost returns one.
//
// Frost returns every aggregate through a hidden out-pointer, uniformly, which
// it is entitled to decide about its own calling convention. C is not uniform: a
// small struct comes back in registers and a large one through a pointer, and
// where the line falls depends on the target and, on some targets, on the field
// types. So calling a C function that returns a struct means classifying the
// return type the way that target's C compiler does, and this is that.
//
// Only what Frost can express is handled: no unions, no bitfields, no packed
// layouts. An enum is the one union-like shape, and the classification rule for
// a union is the same rule applied to every variant at once, which is what
// flattening every variant's fields into one list does.

// One scalar leaf of an aggregate, at its byte offset from the start.
#[derive(Debug, Clone, PartialEq)]
pub struct CScalar {
    pub offset: usize,
    pub ty: Type,
}

// An aggregate flattened to the only two things a C ABI asks about: where the
// bytes are, and which of them are floating point.
#[derive(Debug, Clone, PartialEq)]
pub struct CLayout {
    pub name: String,
    pub size: usize,
    pub align: usize,
    pub scalars: Vec<CScalar>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CTarget {
    // System V AMD64, which is Linux, the BSDs and Intel macOS.
    SysV,
    // Microsoft x64, and the same rule under mingw.
    Windows,
    // AAPCS64, which is arm64 Linux, Apple silicon and Windows on ARM.
    AArch64,
}

// One register the value comes back in, and which slice of the aggregate it
// holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CRegister {
    pub offset: usize,
    pub bytes: usize,
    pub float: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CReturn {
    // The caller allocates the storage and passes its address as a hidden first
    // argument.
    Indirect,
    // The value arrives in these registers, to be written into the caller's
    // storage at the offsets given.
    Registers(Vec<CRegister>),
}

pub fn target_of(triple: &target_lexicon::Triple) -> Option<CTarget> {
    use target_lexicon::{Architecture, OperatingSystem};
    match (triple.architecture, triple.operating_system) {
        (Architecture::X86_64, OperatingSystem::Windows) => {
            Some(CTarget::Windows)
        }
        (Architecture::X86_64, _) => Some(CTarget::SysV),
        (Architecture::Aarch64(_), _) => Some(CTarget::AArch64),
        _ => None,
    }
}

pub fn classify_return(layout: &CLayout, target: CTarget) -> CReturn {
    match target {
        CTarget::Windows => classify_windows(layout),
        CTarget::SysV => classify_sysv(layout),
        CTarget::AArch64 => classify_aarch64(layout),
    }
}

// Microsoft x64: a struct comes back in RAX when its size is 1, 2, 4 or 8
// bytes, and through a hidden pointer otherwise. The contents do not matter,
// which is the part worth stating because it is where this differs from every
// other target here: a `struct { float a; }` comes back in RAX, not XMM0, and a
// `struct { char a[3]; }` goes indirect despite being smaller than a register.
//
// Checked against the host compiler rather than read off a document: gcc on
// this target returns a one-float struct in eax and a three-byte struct through
// a pointer.
fn classify_windows(layout: &CLayout) -> CReturn {
    if matches!(layout.size, 1 | 2 | 4 | 8) {
        return CReturn::Registers(vec![CRegister {
            offset: 0,
            bytes: layout.size,
            float: false,
        }]);
    }
    CReturn::Indirect
}

// System V AMD64: anything over two eightbytes goes in memory. Otherwise each
// eightbyte is SSE when every field touching it is floating point and INTEGER
// when anything else does, so the integer eightbytes come back in RAX and RDX
// and the floating ones in XMM0 and XMM1.
fn classify_sysv(layout: &CLayout) -> CReturn {
    if layout.size > 16 || layout.size == 0 {
        return CReturn::Indirect;
    }
    let count = layout.size.div_ceil(8);
    let mut float = vec![true; count];
    let mut occupied = vec![false; count];
    for scalar in &layout.scalars {
        let size = scalar.ty.size_of().max(1);
        let first = scalar.offset / 8;
        let last = (scalar.offset + size - 1) / 8;
        for eightbyte in first..=last.min(count - 1) {
            occupied[eightbyte] = true;
            if !is_float(&scalar.ty) {
                float[eightbyte] = false;
            }
        }
    }
    let registers = (0..count)
        .map(|eightbyte| CRegister {
            offset: eightbyte * 8,
            bytes: (layout.size - eightbyte * 8).min(8),
            // An eightbyte no field reaches is padding, and padding is not
            // floating point.
            float: occupied[eightbyte] && float[eightbyte],
        })
        .collect();
    CReturn::Registers(registers)
}

// AAPCS64: a homogeneous float aggregate of up to four members comes back in
// the float registers, one member each. Otherwise sixteen bytes or fewer come
// back in x0 and x1, and anything larger goes through a pointer.
fn classify_aarch64(layout: &CLayout) -> CReturn {
    if layout.size == 0 {
        return CReturn::Indirect;
    }
    if let Some(element) = homogeneous_float(layout)
        && layout.scalars.len() <= 4
        && layout.size == layout.scalars.len() * element.size_of()
    {
        return CReturn::Registers(
            layout
                .scalars
                .iter()
                .map(|scalar| CRegister {
                    offset: scalar.offset,
                    bytes: element.size_of(),
                    float: true,
                })
                .collect(),
        );
    }
    if layout.size > 16 {
        return CReturn::Indirect;
    }
    CReturn::Registers(
        (0..layout.size.div_ceil(8))
            .map(|word| CRegister {
                offset: word * 8,
                bytes: (layout.size - word * 8).min(8),
                float: false,
            })
            .collect(),
    )
}

fn homogeneous_float(layout: &CLayout) -> Option<Type> {
    let first = layout.scalars.first()?.ty.clone();
    if !is_float(&first) {
        return None;
    }
    layout
        .scalars
        .iter()
        .all(|scalar| scalar.ty == first)
        .then_some(first)
}

fn is_float(ty: &Type) -> bool {
    matches!(ty, Type::F32 | Type::F64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(size: usize, scalars: &[(usize, Type)]) -> CLayout {
        CLayout {
            name: "T".to_string(),
            size,
            align: 8,
            scalars: scalars
                .iter()
                .map(|(offset, ty)| CScalar {
                    offset: *offset,
                    ty: ty.clone(),
                })
                .collect(),
        }
    }

    fn integer(offset: usize, bytes: usize) -> CRegister {
        CRegister {
            offset,
            bytes,
            float: false,
        }
    }

    fn floating(offset: usize, bytes: usize) -> CRegister {
        CRegister {
            offset,
            bytes,
            float: true,
        }
    }

    // Every one of these was read off the host C compiler's output, which is
    // the only reason to believe them. The float cases are the interesting
    // ones: a struct of one float comes back in an integer register here, which
    // is what makes this target different from the other two.
    #[test]
    fn windows_returns_power_of_two_sizes_in_one_register() {
        let cases: &[(usize, CReturn)] = &[
            (1, CReturn::Registers(vec![integer(0, 1)])),
            (2, CReturn::Registers(vec![integer(0, 2)])),
            (3, CReturn::Indirect),
            (4, CReturn::Registers(vec![integer(0, 4)])),
            (8, CReturn::Registers(vec![integer(0, 8)])),
            (16, CReturn::Indirect),
        ];
        for (size, expected) in cases {
            let scalars: Vec<(usize, Type)> = vec![(0, Type::U8)];
            let found =
                classify_return(&layout(*size, &scalars), CTarget::Windows);
            assert_eq!(&found, expected, "size {size}");
        }
    }

    #[test]
    fn windows_returns_a_float_struct_in_an_integer_register() {
        assert_eq!(
            classify_return(&layout(4, &[(0, Type::F32)]), CTarget::Windows),
            CReturn::Registers(vec![integer(0, 4)])
        );
    }

    #[test]
    fn sysv_splits_into_eightbytes_by_content() {
        assert_eq!(
            classify_return(&layout(8, &[(0, Type::I64)]), CTarget::SysV),
            CReturn::Registers(vec![integer(0, 8)])
        );
        assert_eq!(
            classify_return(
                &layout(16, &[(0, Type::I64), (8, Type::I64)]),
                CTarget::SysV
            ),
            CReturn::Registers(vec![integer(0, 8), integer(8, 8)])
        );
        assert_eq!(
            classify_return(
                &layout(8, &[(0, Type::F32), (4, Type::F32)]),
                CTarget::SysV
            ),
            CReturn::Registers(vec![floating(0, 8)])
        );
        // Mixed within one eightbyte is INTEGER, and the classes are per
        // eightbyte rather than per struct.
        assert_eq!(
            classify_return(
                &layout(16, &[(0, Type::I32), (4, Type::F32), (8, Type::F64)]),
                CTarget::SysV
            ),
            CReturn::Registers(vec![integer(0, 8), floating(8, 8)])
        );
        assert_eq!(
            classify_return(
                &layout(24, &[(0, Type::I64), (8, Type::I64), (16, Type::I64)]),
                CTarget::SysV
            ),
            CReturn::Indirect
        );
        // Three bytes is one partial eightbyte, unlike on Windows.
        assert_eq!(
            classify_return(
                &layout(3, &[(0, Type::U8), (1, Type::U8), (2, Type::U8)]),
                CTarget::SysV
            ),
            CReturn::Registers(vec![integer(0, 3)])
        );
    }

    #[test]
    fn aarch64_returns_a_homogeneous_float_aggregate_in_float_registers() {
        assert_eq!(
            classify_return(
                &layout(
                    16,
                    &[
                        (0, Type::F32),
                        (4, Type::F32),
                        (8, Type::F32),
                        (12, Type::F32)
                    ]
                ),
                CTarget::AArch64
            ),
            CReturn::Registers(vec![
                floating(0, 4),
                floating(4, 4),
                floating(8, 4),
                floating(12, 4)
            ])
        );
        // Five of them is no longer homogeneous enough, and twenty bytes is
        // over the limit, so it goes indirect.
        assert_eq!(
            classify_return(
                &layout(
                    20,
                    &[
                        (0, Type::F32),
                        (4, Type::F32),
                        (8, Type::F32),
                        (12, Type::F32),
                        (16, Type::F32)
                    ]
                ),
                CTarget::AArch64
            ),
            CReturn::Indirect
        );
        assert_eq!(
            classify_return(
                &layout(16, &[(0, Type::I64), (8, Type::I64)]),
                CTarget::AArch64
            ),
            CReturn::Registers(vec![integer(0, 8), integer(8, 8)])
        );
        assert_eq!(
            classify_return(
                &layout(24, &[(0, Type::I64), (8, Type::I64), (16, Type::I64)]),
                CTarget::AArch64
            ),
            CReturn::Indirect
        );
    }
}
