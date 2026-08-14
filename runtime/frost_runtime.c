/* This is compiled as `-std=c11`, which is strict ISO C, and under it a POSIX
   library declares only what ISO C has: no `siginfo_t`, no `stack_t`, no
   `sigaltstack`, no `write`. Asking for POSIX is what puts them back, and it has
   to be asked before the first header is read, so it sits above every include
   rather than beside the ones that need it.

   POSIX is not one level, and the piece this runtime needs is above the base.
   `sigaltstack`, `stack_t` and `SA_ONSTACK` are X/Open rather than base POSIX,
   and glibc keeps them behind `_XOPEN_SOURCE` however recent a
   `_POSIX_C_SOURCE` it is given. Naming only the latter left the alternate
   stack the fault handler runs on undeclared, which is a compile error rather
   than a handler that quietly does nothing.

   Apple inverts the relationship: naming `_POSIX_C_SOURCE` there *narrows*
   what `<signal.h>` offers, so the same request that widens glibc would hide
   `SA_ONSTACK`, and `_DARWIN_C_SOURCE` is what puts it back.

   Left alone wherever the toolchain has already chosen a level, since raising
   one it picked is its call to make. */
#if !defined(_WIN32)
#if !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 700
#endif
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#endif

#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#else
/* `signal.h` and `unistd.h` belong here rather than beside the stack handler
   that wants them. The handler is written further down than the first use of
   `write`, so including them there left that call with no declaration at all
   and the one the header then gave it conflicted with the guess. */
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__has_include)
#if __has_include(<execinfo.h>)
#define FROST_HAS_EXECINFO 1
#include <execinfo.h>
#endif
#endif
#endif

/* What was running when a check failed.

   A bounds check that says only "index 7 out of bounds for length 4" tells you
   what happened and not where, and where is the whole question: the same
   container is indexed from twenty places. There is no debug information yet,
   so these are addresses rather than names, but an address plus the binary is
   enough for addr2line, and it is the difference between a bug you can find
   and a number you cannot. */
static void frost_rt_backtrace(void) {
#if defined(_WIN32)
    void *frames[32];
    USHORT taken = CaptureStackBackTrace(1, 32, frames, 0);
    if (taken == 0) {
        return;
    }
    fprintf(stderr, "frost: called from\n");
    for (USHORT i = 0; i < taken; i++) {
        fprintf(stderr, "  %p\n", frames[i]);
    }
#elif defined(FROST_HAS_EXECINFO)
    void *frames[32];
    int taken = backtrace(frames, 32);
    if (taken <= 1) {
        return;
    }
    fprintf(stderr, "frost: called from\n");
    /* The first frame is this function, which the reader already knows. */
    backtrace_symbols_fd(frames + 1, taken - 1, 2);
#endif
}

/* Every check that ends the process goes through here, so the trace is printed
   once and in one place rather than remembered at each call to abort.

   Not static: the checks themselves are written in Frost now, in
   `runtime.frost` beside this, and this is what they end on. Asking the
   platform for a backtrace is the reason it stays here. */
void frost_rt_report_flush(void);

void frost_rt_stop(void) {
    /* `abort` runs no exit handler, so what a run has said is written here or
       not at all. A crash midway through a compile is the case where those
       reports are worth most. */
    frost_rt_report_flush();
    frost_rt_backtrace();
    fflush(stderr);
    abort();
}

/* Running the stack out.

   Every frame wider than a page touches each page on its way down, so the guard
   is hit rather than stepped over and unbounded recursion faults instead of
   writing into whatever is mapped below. That much is the compiler's doing and
   it is what makes this safe. What it is not is legible: the process dies with
   a fault address and nothing saying which of the many ways to fault it was.

   Naming it costs one handler. On Windows the guard raises a distinct exception
   code, so there is nothing to guess. On POSIX it arrives as SIGSEGV like any
   other bad access, so the handler runs on its own stack (the ordinary one is
   what ran out) and says only that the fault is near the stack pointer, which is
   the honest reading: an address a page or two from where the stack was is the
   stack, and one far from it is somebody else's bug. */
static const char frost_rt_stack_message[] =
    "frost: the stack ran out, which is unbounded recursion or a frame too "
    "wide for it\n";

/* Written with the system call rather than with stdio, because a handler runs
   where stdio's own locks and buffers may not be reachable, and on Windows it
   runs on the stack that just ran out. */
static void frost_rt_write_stderr(const char *text, unsigned long length) {
#if defined(_WIN32)
    DWORD written = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), text, length, &written, NULL);
#else
    ssize_t ignored = write(2, text, (size_t)length);
    (void)ignored;
#endif
}

#if defined(_WIN32)
static LONG CALLBACK frost_rt_stack_handler(EXCEPTION_POINTERS *info) {
    if (info->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW) {
        frost_rt_write_stderr(frost_rt_stack_message,
                              sizeof(frost_rt_stack_message) - 1);
    }
    /* The message is all this is for. What the process does next is what it
       would have done, so the exit code and the debugger's view are unchanged. */
    return EXCEPTION_CONTINUE_SEARCH;
}

static void frost_rt_install_stack_handler(void) {
    AddVectoredExceptionHandler(1, frost_rt_stack_handler);
}
#else
/* Roughly where the stack was when the program started, and how far below it a
   fault still counts as the stack running out. The window is generous on
   purpose: mislabelling one crash costs a wrong sentence, while missing the
   common case costs the whole point of the handler. Anything outside it is
   somebody else's bad pointer and is left to report itself. */
static char *frost_rt_stack_top = NULL;
#define FROST_RT_STACK_WINDOW (256 * 1024 * 1024)

static char frost_rt_handler_stack[64 * 1024];

static void frost_rt_stack_handler(int signal, siginfo_t *info, void *context) {
    (void)context;
    char *fault = (char *)info->si_addr;
    if (frost_rt_stack_top != NULL && fault < frost_rt_stack_top
        && fault > frost_rt_stack_top - FROST_RT_STACK_WINDOW) {
        frost_rt_write_stderr(frost_rt_stack_message,
                              sizeof(frost_rt_stack_message) - 1);
    }
    /* Put the default back and return, so the fault happens again and kills the
       process the way it would have. Reporting a different exit code here would
       be the handler changing what the program did rather than explaining it. */
    struct sigaction restore;
    memset(&restore, 0, sizeof(restore));
    restore.sa_handler = SIG_DFL;
    sigemptyset(&restore.sa_mask);
    sigaction(signal, &restore, NULL);
}

static void frost_rt_install_stack_handler(void) {
    stack_t alternate;
    struct sigaction action;
    char here;
    frost_rt_stack_top = &here;
    alternate.ss_sp = frost_rt_handler_stack;
    alternate.ss_size = sizeof(frost_rt_handler_stack);
    alternate.ss_flags = 0;
    /* The handler needs a stack of its own, since the one that ran out is the
       one it would otherwise run on. Without this there is nothing to report
       from and the program dies as it did before, which is why a failure here
       simply leaves the handler uninstalled. */
    if (sigaltstack(&alternate, NULL) != 0) {
        return;
    }
    memset(&action, 0, sizeof(action));
    action.sa_sigaction = frost_rt_stack_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, NULL);
    sigaction(SIGBUS, &action, NULL);
}
#endif

/* Armed before the program's own `main` runs. The generated code's `main` is the
   program's, so there is no earlier place in it to put this, and the loader is
   what runs something ahead of it. */
static void frost_rt_arm_stack_guard(void) {
    frost_rt_install_stack_handler();
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor)) static void frost_rt_arm_at_start(void) {
    frost_rt_arm_stack_guard();
}
#elif defined(_MSC_VER)
#pragma section(".CRT$XCU", read)
static void __cdecl frost_rt_arm_at_start(void) {
    frost_rt_arm_stack_guard();
}
__declspec(allocate(".CRT$XCU")) void(__cdecl *frost_rt_arm_slot)(void) =
    frost_rt_arm_at_start;
#endif

/* A number for one container, so a handle carries which container it came from
   and not only which slot. A generation catches a handle to a slot that has been
   reused; it says nothing about two containers of the same element type and
   capacity, where a slot is in range on both and the generations coincide. They
   coincide most often right after both are reset, when every generation is
   zero, so the case with no protection at all is the one a program starts in.

   The number is a count rather than anything derived from the container, since
   a container is a plain value that may live in a frame, a static or inside
   another struct, and there is no allocation to take an identity from. Seven
   bits, which is what fits above the generation while a packed handle stays
   positive, so two containers reset one after the other always differ and the
   hundred and twenty-eighth comes back round. Never zero: a container whose
   storage is zeroed but which was never reset holds zero, and a handle from a
   reset container must not read against one. */
/* Counted atomically, because a program spawns threads and two of them resetting
   a container each is exactly the case where two containers must not draw the
   same number. A plain increment can read, add and store around another thread
   doing the same, and hand both the number this exists to keep apart. */
static int64_t frost_rt_container_counter = 0;

int64_t frost_rt_container_id(void) {
#if defined(_MSC_VER)
    int64_t drawn = InterlockedIncrement64(
        (volatile LONG64 *)&frost_rt_container_counter);
#else
    int64_t drawn =
        __atomic_add_fetch(&frost_rt_container_counter, 1, __ATOMIC_RELAXED);
#endif
    return (drawn % 127) + 1;
}

/* Emitted output goes to standard output unless a file has been opened for it,
   which is how `-o` works without every emitter having to carry a destination. */
static FILE *frost_rt_emit_target = 0;

/* A backend writes its output in small pieces, hundreds of thousands of them
   for a large program, so the destination has to carry a buffer wide enough
   that those pieces are not each a write to the operating system. A file opened
   here gets one. Standard output does not have one by default on Windows when
   it is a console, and gets one here so that emitting without `-o` costs what
   emitting to a file costs. */
static void frost_rt_emit_buffer(FILE *stream) {
    static char standard_output_buffer[1 << 16];
    if (stream == stdout) {
        setvbuf(stream, standard_output_buffer, _IOFBF,
                sizeof standard_output_buffer);
    }
}

static FILE *frost_rt_emit_where(void) {
    if (frost_rt_emit_target == 0) {
        static int buffered = 0;
        if (buffered == 0) {
            buffered = 1;
            frost_rt_emit_buffer(stdout);
        }
        return stdout;
    }
    return frost_rt_emit_target;
}

/* A destination in memory, which is what a build that assembles its own output
   emits into: the text never becomes a file, so it is neither written nor read
   back. The buffer is owned here and grows to whatever the program emits, so
   there is no size for a caller to guess at and none to run out of. It is kept
   between runs, since a build emitting one unit after another emits about as
   much each time. */
static char *frost_rt_emit_memory = 0;
static int64_t frost_rt_emit_memory_room = 0;
static int64_t frost_rt_emit_memory_used = 0;
static int frost_rt_emit_to_memory_on = 0;

void frost_rt_emit_to_memory(void) {
    frost_rt_emit_to_memory_on = 1;
    frost_rt_emit_memory_used = 0;
}

int64_t frost_rt_emit_memory_length(void) {
    return frost_rt_emit_memory_used;
}

char *frost_rt_emit_memory_at(void) {
    return frost_rt_emit_memory;
}

void frost_rt_emit_memory_done(void) {
    frost_rt_emit_to_memory_on = 0;
}

/* Whether the bytes about to be written go to memory, having made room for
   them. Doubling means the copying costs a constant per byte emitted however
   large the program is. */
static int frost_rt_emit_holds(int64_t length) {
    if (frost_rt_emit_to_memory_on == 0) {
        return 0;
    }
    if (frost_rt_emit_memory_used + length > frost_rt_emit_memory_room) {
        int64_t wanted = frost_rt_emit_memory_room * 2 + length + (1 << 20);
        char *wider = (char *)realloc(frost_rt_emit_memory, (size_t)wanted);
        if (wider == 0) {
            fputs("frost: no room for the emitted text\n", stderr);
            abort();
        }
        frost_rt_emit_memory = wider;
        frost_rt_emit_memory_room = wanted;
    }
    return 1;
}

int64_t frost_rt_emit_open(const char *path) {
    static char file_buffer[1 << 16];
    frost_rt_emit_target = fopen(path, "wb");
    if (frost_rt_emit_target != 0) {
        setvbuf(frost_rt_emit_target, file_buffer, _IOFBF,
                sizeof file_buffer);
    }
    return frost_rt_emit_target != 0;
}

void frost_rt_emit_close(void) {
    if (frost_rt_emit_target != 0) {
        fclose(frost_rt_emit_target);
        frost_rt_emit_target = 0;
    }
}

void frost_rt_emit_str(const char *text) {
    size_t length = strlen(text);
    if (frost_rt_emit_holds((int64_t)length)) {
        memcpy(frost_rt_emit_memory + frost_rt_emit_memory_used, text, length);
        frost_rt_emit_memory_used += (int64_t)length;
        return;
    }
    fputs(text, frost_rt_emit_where());
}

/* Emit a counted run of bytes rather than a NUL-terminated string, so the
   caller passes a length-carrying `str` and the read is bounded by it. This is
   what lets the emit path be safe. Nothing scans for a terminator. */
void frost_rt_emit_bytes(const char *data, int64_t length) {
    if (frost_rt_emit_holds(length)) {
        memcpy(frost_rt_emit_memory + frost_rt_emit_memory_used, data,
               (size_t)length);
        frost_rt_emit_memory_used += length;
        return;
    }
    fwrite(data, 1, (size_t)length, frost_rt_emit_where());
}

/* A backend emits a number for every slot, label and immediate it writes, which
   is hundreds of thousands of numbers for a large program. Going through
   fprintf for each one spends most of the time parsing the same format string
   and taking the stream's lock. The digits themselves are a division loop. */
void frost_rt_emit_int(int64_t value) {
    char digits[24];
    int at = (int)sizeof digits;
    /* Negating the most negative value overflows, so the magnitude is taken
       one step away from the edge and put back. */
    uint64_t magnitude = value < 0
        ? (uint64_t)(-(value + 1)) + 1u
        : (uint64_t)value;
    do {
        digits[--at] = (char)('0' + (magnitude % 10u));
        magnitude /= 10u;
    } while (magnitude != 0);
    if (value < 0) {
        digits[--at] = '-';
    }
    size_t length = (size_t)((int)sizeof digits - at);
    if (frost_rt_emit_holds((int64_t)length)) {
        memcpy(frost_rt_emit_memory + frost_rt_emit_memory_used, digits + at,
               length);
        frost_rt_emit_memory_used += (int64_t)length;
        return;
    }
    fwrite(digits + at, 1, length, frost_rt_emit_where());
}

void frost_rt_emit_char(int64_t byte) {
    if (frost_rt_emit_holds(1)) {
        frost_rt_emit_memory[frost_rt_emit_memory_used] = (char)byte;
        frost_rt_emit_memory_used += 1;
        return;
    }
    fputc((int)byte, frost_rt_emit_where());
}

/* What std/io.frost writes through. A whole formatted line arrives in one
   call, since the line is composed before any of it leaves, so this is one
   write per `print` rather than one per value. Pinned to stdout rather than
   the emit target, so a program that redirects the compiler's emitted text
   still prints its own output where a reader looks for it. */
void frost_rt_write_bytes(const char *data, int64_t length) {
    fwrite(data, 1, (size_t)length, stdout);
}

/* Digits and one byte, for a program that has no library to reach for. The
   self-hosted compiler's built-in program is one of these: it imports nothing,
   so it declares these itself and writes with them, which is what makes it a
   test that the compiler works with nothing else present. std/io.frost
   composes its own digits and does not call these. */
void frost_rt_write_i64(int64_t value) {
    printf("%lld", (long long)value);
}

void frost_rt_write_char(int64_t byte) {
    fputc((int)byte, stdout);
}

/* A float spelled the way C writes %g, into a buffer rather than out to a
   stream, so the line it belongs to can be finished before anything is
   written. Shortest-round-trip decimal is the one piece of formatting Frost
   does not do for itself: writing it again here would be a second
   implementation of the C library's, differing only where it was wrong.

   Answers how many bytes were written, which is never more than `room`.
   snprintf answers what it would have written, so a value that does not fit
   is reported as the truncated length rather than the wanted one. */
int64_t frost_rt_format_f64(double value, char *out, int64_t room) {
    int wanted;
    if (room <= 0) {
        return 0;
    }
    wanted = snprintf(out, (size_t)room, "%g", value);
    if (wanted < 0) {
        return 0;
    }
    if ((int64_t)wanted >= room) {
        return room - 1;
    }
    return (int64_t)wanted;
}

const char *frost_rt_getenv(const char *name) {
    const char *value = getenv(name);
    if (value == 0) {
        return "";
    }
    return value;
}

/* Set a variable in this process's own environment, which is what a command
   run through `system` inherits. There is no way to hand a child one variable
   without going through here, since the shell line is the only other channel
   and quoting an assignment onto it differs per platform. Answers whether it
   took. */
int64_t frost_rt_setenv(const char *name, const char *value) {
#if defined(_WIN32)
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

const char *frost_rt_read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == 0) {
        return "";
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *)malloc((size_t)length + 1);
    size_t read = fread(buffer, 1, (size_t)length, file);
    buffer[read] = 0;
    fclose(file);
    return buffer;
}

/* The test runner. A failing assertion has to end the test it is in without
   ending the run, or one bad test hides every test after it. The escape is a
   longjmp back into frost_rt_test_run, which is why the runner takes the test body
   as a function pointer rather than being a sequence the compiler emits. The
   setjmp has to own the call. */
/* On Win64 longjmp unwinds through SEH, which needs unwind information for
   every frame it passes. The assembly backend emits none, so a test body that
   fails an assertion would fault on the way out rather than escaping. Setting
   the jump with no frame makes longjmp a plain register restore, which is all
   that escaping a hand-written frame needs. */
#if defined(_WIN32) && defined(__GNUC__)
#define frost_rt_setjmp(env) _setjmp((env), 0)
#else
#define frost_rt_setjmp(env) setjmp(env)
#endif

static jmp_buf frost_rt_test_escape;
static int frost_rt_inside_test = 0;
static int64_t frost_rt_tests_passed = 0;
static int64_t frost_rt_tests_failed = 0;

void frost_rt_test_run(const char *name, void (*body)(void)) {
    printf("test %s ... ", name);
    fflush(stdout);
    frost_rt_inside_test = 1;
    if (frost_rt_setjmp(frost_rt_test_escape) == 0) {
        body();
        frost_rt_tests_passed++;
        printf("ok\n");
    } else {
        frost_rt_tests_failed++;
    }
    frost_rt_inside_test = 0;
    fflush(stdout);
}

/* Returns the failure count, so the process can exit non-zero on it. */
int64_t frost_rt_test_summary(void) {
    printf("\n%lld passed, %lld failed\n", (long long)frost_rt_tests_passed,
           (long long)frost_rt_tests_failed);
    fflush(stdout);
    return frost_rt_tests_failed;
}

/* Reports as JSON. While a record is open every piece of a message is held
   rather than written, so the whole message can go out as one JSON string when
   the record closes. A place opens a record; whatever ends the report closes
   it. */
static int frost_rt_json_on = 0;
static int frost_rt_json_open = 0;
static char *frost_rt_json_message = 0;
static size_t frost_rt_json_length = 0;
static size_t frost_rt_json_room = 0;
static char frost_rt_json_file[1024];
static int64_t frost_rt_json_line = 0;
static int64_t frost_rt_json_column = 0;
static int64_t frost_rt_json_offset = 0;

void frost_rt_json_reports(void) { frost_rt_json_on = 1; }

static void frost_rt_json_hold(const char *data, size_t length) {
    if (frost_rt_json_length + length + 1 > frost_rt_json_room) {
        size_t room = frost_rt_json_room * 2 + length + 256;
        char *grown = (char *)realloc(frost_rt_json_message, room);
        if (grown == 0) {
            return;
        }
        frost_rt_json_message = grown;
        frost_rt_json_room = room;
    }
    memcpy(frost_rt_json_message + frost_rt_json_length, data, length);
    frost_rt_json_length += length;
}

/* The report buffer below, which a closed record is written into rather than
   written out, so JSON records come out in the order they are written the same
   as the block form does. */
static void frost_rt_report_begin(int64_t at);
static int frost_rt_report_append(const char *data, size_t length);

static void frost_rt_json_put(const char *text) {
    frost_rt_report_append(text, strlen(text));
}

static void frost_rt_json_putc(char byte) {
    frost_rt_report_append(&byte, 1);
}

/* Holds the open record, if there is one, as one object on one line. */
void frost_rt_json_close(void) {
    if (!frost_rt_json_on || (!frost_rt_json_open && frost_rt_json_length == 0)) {
        return;
    }
    frost_rt_json_open = 0;
    frost_rt_report_begin(frost_rt_json_offset);
    frost_rt_json_put("{\"file\":\"");
    for (const char *at = frost_rt_json_file; *at; at++) {
        if (*at == '\\' || *at == '\"') {
            frost_rt_json_putc('\\');
        }
        frost_rt_json_putc(*at);
    }
    char numbers[128];
    int written = snprintf(numbers, sizeof(numbers),
                           "\",\"line\":%lld,\"column\":%lld,"
                           "\"span\":[%lld,%lld],\"severity\":\"error\","
                           "\"message\":\"",
                           (long long)frost_rt_json_line,
                           (long long)frost_rt_json_column,
                           (long long)frost_rt_json_offset,
                           (long long)frost_rt_json_offset);
    if (written > 0) {
        frost_rt_report_append(numbers, (size_t)written);
    }
    for (size_t at = 0; at < frost_rt_json_length; at++) {
        unsigned char held = (unsigned char)frost_rt_json_message[at];
        if (held == '\"' || held == '\\') {
            frost_rt_json_putc('\\');
            frost_rt_json_putc((char)held);
        } else if (held == '\n') {
            frost_rt_json_put("\\n");
        } else if (held < 0x20) {
            char escaped[8];
            int room = snprintf(escaped, sizeof(escaped), "\\u%04x", held);
            if (room > 0) {
                frost_rt_report_append(escaped, (size_t)room);
            }
        } else {
            frost_rt_json_putc((char)held);
        }
    }
    frost_rt_json_put("\"}\n");
    frost_rt_json_length = 0;
}

/* Opens a record. Everything written after this belongs to its message. */
void frost_rt_json_place(const char *path, int64_t line, int64_t column,
                         int64_t offset) {
    if (!frost_rt_json_on) {
        return;
    }
    frost_rt_json_close();
    size_t held = strlen(path);
    if (held >= sizeof(frost_rt_json_file)) {
        held = sizeof(frost_rt_json_file) - 1;
    }
    memcpy(frost_rt_json_file, path, held);
    frost_rt_json_file[held] = 0;
    frost_rt_json_line = line;
    frost_rt_json_column = column;
    frost_rt_json_offset = offset;
    frost_rt_json_open = 1;
}

/* Parser recovery for the self-hosted compiler. A parse fault has to end
   the declaration it is in without ending the compile, or one bad
   declaration hides every diagnostic after it. The escape is the same
   longjmp the test runner uses, with the same Win64 rule: the setjmp has to
   own the call, so the runner takes the loop body as a function pointer and
   its context the way frost_rt_test_run takes a test body. Marks nest,
   since a block's statement loop recovers inside a declaration's, so the
   environments live on a small stack. */
#define FROST_RT_RECOVER_DEPTH 32
static jmp_buf frost_rt_recover_stack[FROST_RT_RECOVER_DEPTH];
static int frost_rt_recover_depth = 0;

void frost_rt_die(void);
/* Ends the line a report was written on, into the report where one is open. */
void frost_rt_end_line(void);

/* Runs the body with a mark armed. Answers 0 when the body returned and 1
   when it escaped. */
int64_t frost_rt_recover_run(void (*body)(char *), char *context) {
    if (frost_rt_recover_depth >= FROST_RT_RECOVER_DEPTH) {
        /* Deeper than any real nesting; running unprotected would turn a
           fault into an abort, so say what happened first. */
        fprintf(stderr, "frost: recovery marks nested past %d\n",
                FROST_RT_RECOVER_DEPTH);
        fflush(stderr);
        frost_rt_stop();
    }
    if (frost_rt_setjmp(frost_rt_recover_stack[frost_rt_recover_depth]) ==
        0) {
        frost_rt_recover_depth++;
        body(context);
        frost_rt_recover_depth--;
        return 0;
    }
    return 1;
}

static int64_t frost_rt_recover_count = 0;

/* Ends the enclosing frost_rt_recover_run. With no mark armed a fault has
   nowhere to escape to, so it ends the process the way it always did. The
   count is what lets a compile that recovered still refuse at the end:
   recovery reports more, it never accepts more. */
void frost_rt_recover_escape(void) {
    if (frost_rt_recover_depth == 0) {
        frost_rt_die();
    }
    if (frost_rt_json_on) {
        frost_rt_json_close();
        frost_rt_recover_count++;
        frost_rt_recover_depth--;
        longjmp(frost_rt_recover_stack[frost_rt_recover_depth], 1);
    }
    /* The newline frost_rt_die would have written, so a report composed
       piece by piece ends its line when the parse goes on instead. */
    frost_rt_end_line();
    frost_rt_recover_count++;
    frost_rt_recover_depth--;
    longjmp(frost_rt_recover_stack[frost_rt_recover_depth], 1);
}

/* A fault a walk can carry on past. Some checks hold nothing across the thing
   they are checking: a call is checked against the signature it names and the
   next call is checked against its own, so a fault in one leaves nothing
   half-written for the next to trip over. Those report and keep walking, which
   needs the count and the newline that end a report without the escape. Arming
   a mark per call would be a setjmp per node of the program, which is what this
   is instead of. */
void frost_rt_recover_note(void) {
    if (frost_rt_json_on) {
        frost_rt_json_close();
        frost_rt_recover_count++;
        return;
    }
    frost_rt_end_line();
    frost_rt_recover_count++;
}

int64_t frost_rt_recover_failures(void) { return frost_rt_recover_count; }

/* An assertion outside a test has nowhere to escape to, so it still aborts.
   Inside one it fails that test and the run carries on. */
static void frost_rt_assert_failed(const char *where) {
    printf("FAILED\n");
    fflush(stdout);
    if (where != 0) {
        fprintf(stderr, "  assertion failed at %s\n", where);
    } else {
        fprintf(stderr, "  assertion failed\n");
    }
    fflush(stderr);
    if (frost_rt_inside_test) {
        longjmp(frost_rt_test_escape, 1);
    }
    frost_rt_stop();
}

void frost_rt_assert(int8_t condition) {
    if (!condition) {
        frost_rt_assert_failed(0);
    }
}

/* The same assertion carrying the source position the compiler knew, so a
   failure names the line the reader wrote rather than only the test it was in. */
void frost_rt_assert_at(int8_t condition, const char *where) {
    if (!condition) {
        frost_rt_assert_failed(where);
    }
}

/* Diagnostics for a Frost-written compiler. Its program output goes to stdout,
   so errors are composed piecewise on stderr and frost_rt_die ends the process. */
void frost_rt_error(const char *text) {
    fputs(text, stderr);
}

/* One byte, straight to stderr. The checks in runtime.frost compose their
   numbers a digit at a time through this, which is what lets them format
   without a buffer: a buffer is indexed, and an index is what those checks are
   there to answer for. Straight, rather than through frost_rt_error_int, because
   a check that failed is not a report and must not be held back while reports
   are being collected as JSON. */
/* Reports held until the run is over, so they go out in the order they are
   written rather than the order the passes found them. Which pass looks at a
   file first is the compiler's business; a reader reads top to bottom.

   A report opens with the byte its place is at and takes every piece written
   after it. Anything written with none open goes straight out: a line about the
   command rather than about the program has no place to sort against. */
typedef struct {
    /* Where the report sorts. A second line of one report carries the first
       line's place rather than its own, so the two stay together and the pair
       sorts where the fault is rather than where the note points. */
    int64_t at;
    char *text;
    size_t length;
    size_t room;
} frost_rt_report;

static frost_rt_report *frost_rt_reports = 0;
static size_t frost_rt_report_count = 0;
static size_t frost_rt_report_room = 0;
static int frost_rt_report_armed = 0;

void frost_rt_report_flush(void);

/* Everything held, in the order it is written, then nothing held. Runs at exit,
   which covers a refused run and one that only warned, and can be called again
   by hand without writing anything twice. */
void frost_rt_report_flush(void) {
    size_t count = frost_rt_report_count;
    /* Cleared first: writing a report cannot open another, and a second flush
       finding the list empty is what makes calling it twice harmless. */
    frost_rt_report_count = 0;
    /* An insertion sort, which is stable, so two reports about one place stay
       in the order they were found. There are as many of these as a run has
       faults, and a run with enough for the order to cost anything has already
       told the reader more than they will read. */
    for (size_t outer = 1; outer < count; outer++) {
        frost_rt_report held = frost_rt_reports[outer];
        size_t inner = outer;
        while (inner > 0 && frost_rt_reports[inner - 1].at > held.at) {
            frost_rt_reports[inner] = frost_rt_reports[inner - 1];
            inner--;
        }
        frost_rt_reports[inner] = held;
    }
    for (size_t index = 0; index < count; index++) {
        frost_rt_report *held = &frost_rt_reports[index];
        if (held->length > 0) {
            fwrite(held->text, 1, held->length, stderr);
        }
        free(held->text);
        held->text = 0;
        held->length = 0;
        held->room = 0;
    }
    fflush(stderr);
}

static void frost_rt_report_begin(int64_t at) {
    if (!frost_rt_report_armed) {
        frost_rt_report_armed = 1;
        atexit(frost_rt_report_flush);
    }
    if (frost_rt_report_count == frost_rt_report_room) {
        size_t room = frost_rt_report_room * 2 + 16;
        frost_rt_report *grown = (frost_rt_report *)realloc(
            frost_rt_reports, room * sizeof(frost_rt_report));
        if (grown == 0) {
            return;
        }
        frost_rt_reports = grown;
        frost_rt_report_room = room;
    }
    frost_rt_report *held = &frost_rt_reports[frost_rt_report_count++];
    held->at = at;
    held->text = 0;
    held->length = 0;
    held->room = 0;
}

/* Opens a report at a place. Everything written after this belongs to it. */
void frost_rt_report_open(int64_t at) {
    /* A JSON record opens its own report when it closes, holding the object it
       renders to rather than the pieces written into it, so the place is taken
       there instead of here. */
    if (frost_rt_json_on) {
        return;
    }
    frost_rt_report_begin(at);
}

/* Opens another line of the report already open, at a place of its own. It
   sorts with the report it continues, so a fault and the note under it are one
   thing to read wherever each of them points. */
void frost_rt_report_note(void) {
    if (frost_rt_json_on) {
        return;
    }
    int64_t held =
        frost_rt_report_count == 0
            ? 0
            : frost_rt_reports[frost_rt_report_count - 1].at;
    frost_rt_report_open(held);
}

/* Appends to the open report, or answers 0 where there is none to append to. */
static int frost_rt_report_append(const char *data, size_t length) {
    if (frost_rt_report_count == 0) {
        return 0;
    }
    frost_rt_report *held = &frost_rt_reports[frost_rt_report_count - 1];
    if (held->length + length + 1 > held->room) {
        size_t room = held->room * 2 + length + 256;
        char *grown = (char *)realloc(held->text, room);
        if (grown == 0) {
            return 0;
        }
        held->text = grown;
        held->room = room;
    }
    memcpy(held->text + held->length, data, length);
    held->length += length;
    return 1;
}

static int frost_rt_report_hold(const char *data, size_t length) {
    if (frost_rt_json_on) {
        return 0;
    }
    return frost_rt_report_append(data, length);
}

/* The end of the line a report was written on. It belongs to that report, so
   it is held with the rest of it; with none open it goes straight out, which is
   what ends a line the run wrote about itself. */
void frost_rt_end_line(void) {
    char held = '\n';
    if (frost_rt_report_hold(&held, 1)) {
        return;
    }
    fputc((int)held, stderr);
    fflush(stderr);
}

void frost_rt_error_char(int64_t byte) {
    char held = (char)(unsigned char)byte;
    if (frost_rt_report_hold(&held, 1)) {
        return;
    }
    fputc((int)(unsigned char)byte, stderr);
}

/* Write a counted run of bytes to stderr, so a diagnostic composed from a `str`
   is bounded by the length it carries rather than a NUL. */
void frost_rt_error_bytes(const char *data, int64_t length) {
    if (frost_rt_json_on) {
        frost_rt_json_hold(data, (size_t)length);
        return;
    }
    if (frost_rt_report_hold(data, (size_t)length)) {
        return;
    }
    fwrite(data, 1, (size_t)length, stderr);
}

void frost_rt_error_src(const char *text, int64_t offset, int64_t length) {
    if (frost_rt_json_on) {
        frost_rt_json_hold(text + offset, (size_t)length);
        return;
    }
    if (frost_rt_report_hold(text + offset, (size_t)length)) {
        return;
    }
    fwrite(text + offset, 1, (size_t)length, stderr);
}

void frost_rt_error_int(int64_t value) {
    char held[32];
    int written = snprintf(held, sizeof(held), "%lld", (long long)value);
    if (written <= 0) {
        return;
    }
    if (frost_rt_json_on) {
        frost_rt_json_hold(held, (size_t)written);
        return;
    }
    if (frost_rt_report_hold(held, (size_t)written)) {
        return;
    }
    fwrite(held, 1, (size_t)written, stderr);
}

void frost_rt_die(void) {
    if (frost_rt_json_on) {
        frost_rt_json_close();
        frost_rt_report_flush();
        exit(1);
    }
    /* The line the last report was written on, into that report, and then
       everything held goes out in the order it is written. */
    frost_rt_end_line();
    frost_rt_report_flush();
    exit(1);
}

/* The command line, reached without the emitted program's `main` having to
   carry it. A Frost `main` takes no parameters and both backends emit it that
   way, so the arguments are captured here instead: from the C runtime's own
   copies on Windows, and from the initializer glibc and macOS hand argc and
   argv to everywhere else. */
#if defined(_WIN32)
#include <stdlib.h>
static int frost_rt_argument_count(void) {
    return __argc;
}
static char **frost_rt_argument_vector(void) {
    return __argv;
}
#else
static int frost_rt_saved_argc = 0;
static char **frost_rt_saved_argv = 0;

__attribute__((constructor)) static void frost_rt_capture_arguments(int argc,
                                                                char **argv) {
    frost_rt_saved_argc = argc;
    frost_rt_saved_argv = argv;
}

static int frost_rt_argument_count(void) {
    return frost_rt_saved_argc;
}
static char **frost_rt_argument_vector(void) {
    return frost_rt_saved_argv;
}
#endif

int64_t frost_rt_arg_count(void) {
    return (int64_t)frost_rt_argument_count();
}

/* Out of range answers with the empty string rather than failing, so a caller
   reads arguments by asking rather than by counting first. */
const char *frost_rt_arg_at(int64_t index) {
    if (index < 0 || index >= (int64_t)frost_rt_argument_count()) {
        return "";
    }
    return frost_rt_argument_vector()[index];
}

/* Heap allocation for the standard library's growable containers. Thin wrappers
   so a Frost program names one set of functions rather than the C library's,
   and so a freestanding build can point them at its own allocator. */
/* How many blocks are out. A container that frees what it took brings this back
   to where it found it, which is what lets a test say so: a leak is otherwise
   invisible until a program runs long enough to notice. Counting is two
   increments on a path that already calls malloc, so it costs nothing worth
   measuring. */
static int64_t frost_rt_heap_blocks = 0;

int64_t frost_rt_heap_live(void) {
    return frost_rt_heap_blocks;
}

/* An allocation that fails aborts rather than answering with nothing, for the
   reason `frost_rt_alloc` in runtime.frost says. These two stay here because
   they count the blocks that are out, and a count is a static, which Frost has
   no way to declare. */
void *frost_rt_heap_alloc(int64_t size) {
    void *block = malloc((size_t)size);
    if (block == NULL) {
        fprintf(stderr,
                "frost: out of memory asking for %lld bytes\n",
                (long long)size);
        frost_rt_stop();
    }
    frost_rt_heap_blocks += 1;
    return block;
}

void *frost_rt_heap_realloc(void *block, int64_t size) {
    void *moved = realloc(block, (size_t)size);
    if (moved == NULL) {
        fprintf(stderr,
                "frost: out of memory asking for %lld bytes\n",
                (long long)size);
        frost_rt_stop();
    }
    if (block == NULL) {
        frost_rt_heap_blocks += 1;
    }
    return moved;
}

void frost_rt_heap_free(void *block) {
    if (block != NULL) {
        frost_rt_heap_blocks -= 1;
    }
    free(block);
}

/* Whole-file read and write, for a standard library that does its own IO
   rather than reaching for the C library directly. The read returns a fresh
   heap block the caller frees. The length comes back through frost_rt_file_size. */
static int64_t frost_rt_last_read_length = 0;

const char *frost_rt_file_read(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == 0) {
        frost_rt_last_read_length = -1;
        /* A block rather than a literal. The caller frees what it was handed
           whether the read worked or not, and handing back a literal makes
           that free a program giving back storage it never took. */
        char *empty = (char *)frost_rt_heap_alloc(1);
        empty[0] = 0;
        return empty;
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    /* Counted, because `fs_free` gives it back through `frost_rt_heap_free`,
       which counts. A block allocated one way and freed the other leaves
       `heap_live` one lower than it started, so a program that reads a file
       reports a leak somewhere it does not have one and hides one it does. */
    char *buffer = (char *)frost_rt_heap_alloc(length + 1);
    size_t read = fread(buffer, 1, (size_t)length, file);
    buffer[read] = 0;
    fclose(file);
    frost_rt_last_read_length = (int64_t)read;
    return buffer;
}

// The seconds on the wall, which is what names one session's log file apart
// from the last one's. A count rather than a date: what a name has to do here is
// differ, and a program that wants it readable formats it itself.
int64_t frost_rt_wall_seconds(void) {
    return (int64_t)time(NULL);
}

int64_t frost_rt_file_size(void) {
    return frost_rt_last_read_length;
}

int64_t frost_rt_file_write(const char *path, const char *bytes, int64_t length) {
    FILE *file = fopen(path, "wb");
    if (file == 0) {
        return 0;
    }
    size_t written = fwrite(bytes, 1, (size_t)length, file);
    fclose(file);
    return written == (size_t)length;
}

int64_t frost_rt_file_exists(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == 0) {
        return 0;
    }
    fclose(file);
    return 1;
}

/* A path with the working directory in front of it where it had none, so two
   spellings of one file compare equal. A relative path and an absolute one name
   the same file and share no prefix, which is what makes this the only way to
   ask whether one path sits under another. A path that cannot be resolved comes
   back as it was given, since the caller's next step says so better. */
char *frost_rt_absolute_path(const char *path) {
#if defined(_WIN32)
    char *resolved = _fullpath(0, path, 0);
#else
    char *resolved = realpath(path, 0);
#endif
    if (resolved != 0) {
        return resolved;
    }
    size_t length = strlen(path);
    char *copy = (char *)malloc(length + 1);
    if (copy == 0) {
        return (char *)path;
    }
    memcpy(copy, path, length + 1);
    return copy;
}

/* Runs a command line through the shell and answers with its exit status, so
   the compiler can drive the assembler and the linker, and so a `--test` build
   can exit on what the tests said. POSIX `system` encodes the child's exit code
   in the high byte of its return rather than handing it back directly, so a
   caller that returns this value straight out of `main` would see it taken mod
   256 and a failing run report success. Decode it to the plain exit code. */
int64_t frost_rt_run_command(const char *command) {
    int status = system(command);
#if defined(_WIN32)
    return (int64_t)status;
#else
    if (status != -1 && WIFEXITED(status)) {
        return (int64_t)WEXITSTATUS(status);
    }
    return (int64_t)status;
#endif
}

/* OS threads. A spawn takes a function and a context pointer, runs the function
   on a new thread with that pointer, and answers with a handle the caller joins
   later. The context is a `void*` the Frost side gives a type to, the same
   shape a callback uses, so a thread body is an ordinary `fn(mut Ctx)`.

   Windows and POSIX have different thread APIs and different body signatures, so
   each platform wraps the Frost body in a trampoline of its own shape and hands
   the real body and context through a small heap record. */
typedef void (*frost_rt_thread_body)(void *);

struct frost_rt_thread_start {
    frost_rt_thread_body body;
    void *context;
};

#if defined(_WIN32)
#include <windows.h>

static DWORD WINAPI frost_rt_thread_trampoline(LPVOID raw) {
    struct frost_rt_thread_start *start = (struct frost_rt_thread_start *)raw;
    frost_rt_thread_body body = start->body;
    void *context = start->context;
    free(start);
    body(context);
    return 0;
}

int64_t frost_rt_thread_spawn(void *body, void *context) {
    struct frost_rt_thread_start *start =
        (struct frost_rt_thread_start *)malloc(sizeof(*start));
    start->body = (frost_rt_thread_body)body;
    start->context = context;
    HANDLE handle = CreateThread(0, 0, frost_rt_thread_trampoline, start, 0, 0);
    return (int64_t)(intptr_t)handle;
}

void frost_rt_thread_join(int64_t handle) {
    HANDLE h = (HANDLE)(intptr_t)handle;
    WaitForSingleObject(h, INFINITE);
    CloseHandle(h);
}
#else
#include <pthread.h>

static void *frost_rt_thread_trampoline(void *raw) {
    struct frost_rt_thread_start *start = (struct frost_rt_thread_start *)raw;
    frost_rt_thread_body body = start->body;
    void *context = start->context;
    free(start);
    body(context);
    return 0;
}

int64_t frost_rt_thread_spawn(void *body, void *context) {
    struct frost_rt_thread_start *start =
        (struct frost_rt_thread_start *)malloc(sizeof(*start));
    start->body = (frost_rt_thread_body)body;
    start->context = context;
    pthread_t *thread = (pthread_t *)malloc(sizeof(pthread_t));
    pthread_create(thread, 0, frost_rt_thread_trampoline, start);
    return (int64_t)(intptr_t)thread;
}

void frost_rt_thread_join(int64_t handle) {
    pthread_t *thread = (pthread_t *)(intptr_t)handle;
    pthread_join(*thread, 0);
    free(thread);
}
#endif

/* An atomic add, so threads can accumulate into shared storage without a lock.
   Answers the value before the add, like the hardware primitive. */
int64_t frost_rt_atomic_add_i64(void *cell, int64_t amount) {
#if defined(_WIN32)
    return (int64_t)InterlockedExchangeAdd64((volatile long long *)cell,
                                             (long long)amount);
#else
    return __sync_fetch_and_add((int64_t *)cell, amount);
#endif
}

/* Which calling convention the native backend must emit for. */
int64_t frost_rt_is_windows(void) {
#ifdef _WIN32
    return 1;
#else
    return 0;
#endif
}

// Which of the three the build is for, as 0, 1 and 2. A compiler reads it to
// settle the target constants a `when` chooses on, and the two compilers have
// to answer alike, so the answer comes from one place rather than from each
// asking its own host in its own way.
int64_t frost_rt_target_os(void) {
#ifdef _WIN32
    return 0;
#elif defined(__APPLE__)
    return 2;
#else
    return 1;
#endif
}
