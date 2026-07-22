import io

p = 'bootstrap/frost.frost'
s = io.open(p, encoding='utf-8').read()
NL = chr(92) + 'n'


def sub(old, new, why):
    global s
    if old not in s:
        raise SystemExit('not found: ' + why)
    s = s.replace(old, new, 1)


sub('frost_die       :: extern fn()',
    'frost_die       :: extern fn()\nfrost_is_windows :: extern fn() -> i64',
    'extern')

sub('    label_id: i64,', '    label_id: i64,\n    abi_windows: i64,', 'field')
sub('        label_id = 0,', '        label_id = 0,\n        abi_windows = frost_is_windows(),', 'init')

# Two calling conventions. Windows x64 passes four integer arguments in
# rcx, rdx, r8, r9 and makes the caller reserve a 32 byte shadow area above
# them. System V passes six in rdi, rsi, rdx, rcx, r8, r9 with no shadow area,
# and wants the count of vector registers in al before a variadic call.
abi = '''abi_arg_regs :: fn(mut p: Parser) -> i64 {
    if (p.abi_windows == 1) {
        return 4
    }
    6
}

abi_shadow :: fn(mut p: Parser) -> i64 {
    if (p.abi_windows == 1) {
        return 32
    }
    0
}

'''
sub('ASM_FRAME :: 4096', abi + 'ASM_FRAME :: 4096', 'abi helpers')

sub('''emit_asm_arg_reg :: fn(index: i64) {
    if (index == 0) {
        frost_emit_str("%rcx")
        return
    }
    if (index == 1) {
        frost_emit_str("%rdx")
        return
    }
    if (index == 2) {
        frost_emit_str("%r8")
        return
    }
    frost_emit_str("%r9")
}''','''emit_asm_arg_reg :: fn(mut p: Parser, index: i64) {
    if (p.abi_windows == 1) {
        if (index == 0) { frost_emit_str("%rcx")  return }
        if (index == 1) { frost_emit_str("%rdx")  return }
        if (index == 2) { frost_emit_str("%r8")   return }
        frost_emit_str("%r9")
        return
    }
    if (index == 0) { frost_emit_str("%rdi")  return }
    if (index == 1) { frost_emit_str("%rsi")  return }
    if (index == 2) { frost_emit_str("%rdx")  return }
    if (index == 3) { frost_emit_str("%rcx")  return }
    if (index == 4) { frost_emit_str("%r8")   return }
    frost_emit_str("%r9")
}''', 'arg reg')

sub('''emit_asm_pop_arg :: fn(index: i64) {
    frost_emit_str("    movq (%rsp), ")
    emit_asm_arg_reg(index)''','''emit_asm_pop_arg :: fn(mut p: Parser, index: i64) {
    frost_emit_str("    movq (%rsp), ")
    emit_asm_arg_reg(p, index)''', 'pop arg')

# The call: registers, then anything past them on the stack above the shadow.
sub('''    total := count + sret
    mut spill : i64 = 0
    if (total > 4) {
        spill = ((total - 4) * 8 + 15) / 16 * 16
    }
    frame := 32 + spill''','''    total := count + sret
    regs := abi_arg_regs(p)
    shadow := abi_shadow(p)
    mut spill : i64 = 0
    if (total > regs) {
        spill = ((total - regs) * 8 + 15) / 16 * 16
    }
    frame := shadow + spill''', 'call frame')

sub('''    if (sret == 1) {
        asm_i("leaq ", result_slot, "(%rbp), %rcx")
    }''','''    if (sret == 1) {
        frost_emit_str("    leaq ")
        frost_emit_int(result_slot)
        frost_emit_str("(%rbp), ")
        emit_asm_arg_reg(p, 0)
        frost_emit_str("@")
    }'''.replace('@', NL), 'sret reg')

sub('''        if (dest < 4) {
            emit_asm_arg_reg(dest)
            frost_emit_str("@")
        } else {
            frost_emit_str("%r10@")
            asm_i("movq %r10, ", 32 + 8 * (dest - 4), "(%rsp)")
        }'''.replace('@', NL),'''        if (dest < regs) {
            emit_asm_arg_reg(p, dest)
            frost_emit_str("@")
        } else {
            frost_emit_str("%r10@")
            asm_i("movq %r10, ", shadow + 8 * (dest - regs), "(%rsp)")
        }'''.replace('@', NL), 'call arg placement')

# System V wants the vector register count in al before a variadic call, and
# every extern here may be one.
sub('''    frost_emit_str("    call ")
    sym : ^Sym = arena_at(p.fns, fn_index)''','''    sym : ^Sym = arena_at(p.fns, fn_index)
    if (p.abi_windows == 0 && sym^.is_extern == 1) {
        asm("movb $0, %al")
    }
    frost_emit_str("    call ")''', 'variadic al')

# The callee finds its stack arguments above the saved frame pointer, the
# return address and whatever shadow area the convention reserves.
sub('''            if (place < 4) {
                frost_emit_str("    movq ")
                emit_asm_arg_reg(place)
                frost_emit_str(", %r11@")
            } else {
                asm_i("movq ", 48 + 8 * (place - 4), "(%rbp), %r11")
            }'''.replace('@', NL),'''            if (place < regs) {
                frost_emit_str("    movq ")
                emit_asm_arg_reg(p, place)
                frost_emit_str(", %r11@")
            } else {
                asm_i("movq ", 16 + shadow + 8 * (place - regs), "(%rbp), %r11")
            }'''.replace('@', NL), 'callee struct param')

sub('''            if (place < 4) {
                frost_emit_str("    movq ")
                emit_asm_arg_reg(place)
                frost_emit_str(", ")
                frost_emit_int(slot)
                frost_emit_str("(%rbp)@")
            } else {
                asm_i("movq ", 48 + 8 * (place - 4), "(%rbp), %r10")
                asm_i("movq %r10, ", slot, "(%rbp)")
            }'''.replace('@', NL),'''            if (place < regs) {
                frost_emit_str("    movq ")
                emit_asm_arg_reg(p, place)
                frost_emit_str(", ")
                frost_emit_int(slot)
                frost_emit_str("(%rbp)@")
            } else {
                asm_i("movq ", 16 + shadow + 8 * (place - regs), "(%rbp), %r10")
                asm_i("movq %r10, ", slot, "(%rbp)")
            }'''.replace('@', NL), 'callee scalar param')

sub('''    mut shift : i64 = 0
    if (is_struct_ty(ret)) {''','''    regs := abi_arg_regs(p)
    shadow := abi_shadow(p)
    mut shift : i64 = 0
    if (is_struct_ty(ret)) {''', 'frame abi locals')

sub('''            frost_emit_str("    movq ")
            emit_asm_arg_reg(place)
            frost_emit_str(", %r11@")'''.replace('@', NL),
    '''            frost_emit_str("    movq ")
            emit_asm_arg_reg(p, place)
            frost_emit_str(", %r11@")'''.replace('@', NL), 'noop')

io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('ok')
