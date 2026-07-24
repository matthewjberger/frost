// A libc-free runtime for `--freestanding` builds. It provides only what the
// generated code needs with no standard library: memcpy for aggregate copies,
// the two safety checks (which trap instead of printing), and a process entry
// point that calls the program's `main` and hands its result to the operating
// system. A program that uses no C functions of its own links against this and
// nothing else, so a fixed or static-arena Frost program reaches a target with
// no libc at all. Output is the process exit code.

typedef long long frost_i64;
typedef unsigned long long frost_usize;

void *memcpy(void *destination, const void *source, frost_usize count) {
    unsigned char *to = (unsigned char *)destination;
    const unsigned char *from = (const unsigned char *)source;
    for (frost_usize i = 0; i < count; i++) {
        to[i] = from[i];
    }
    return destination;
}

void frost_bounds_check(frost_i64 index, frost_i64 length) {
    if ((frost_usize)index >= (frost_usize)length) {
        __builtin_trap();
    }
}

frost_i64 frost_check_index(frost_i64 index, frost_i64 length) {
    frost_bounds_check(index, length);
    return index;
}

void frost_generation_check(frost_i64 stored, frost_i64 expected) {
    if (stored != expected) {
        __builtin_trap();
    }
}

extern int main(void);

// The only platform-specific part: the process entry point, which calls the
// program's `main` and hands its result to the operating system. Each target
// gets its own, so the same feature works on Windows, Linux, and macOS.

#if defined(_WIN32)

// Windows: the OS entry, exiting through kernel32. No C runtime.
__declspec(dllimport) void __stdcall ExitProcess(unsigned int code);

void mainCRTStartup(void) {
    ExitProcess((unsigned int)main());
}

#elif defined(__linux__)

// Linux x86-64: the exit syscall (number 60), no libc.
void _start(void) {
    long code = (long)main();
    long result;
    __asm__ volatile("syscall"
                     : "=a"(result)
                     : "a"(60), "D"(code)
                     : "rcx", "r11", "memory");
    __builtin_unreachable();
}

#elif defined(__APPLE__)

// macOS x86-64: the BSD exit syscall (class 2, number 1). macOS always provides
// its syscall layer through libSystem, so a macOS freestanding build has no C
// standard library but does link libSystem, which is the platform minimum, the
// same floor Rust's macOS targets sit on.
void _start(void) {
    long code = (long)main();
    long result;
    __asm__ volatile("syscall"
                     : "=a"(result)
                     : "a"(0x2000001), "D"(code)
                     : "rcx", "r11", "memory");
    __builtin_unreachable();
}

#else

#error "freestanding builds are not supported on this platform yet"

#endif
