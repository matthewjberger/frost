#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif

void frost_rt_bounds_check(int64_t index, int64_t length) {
    if ((uint64_t)index >= (uint64_t)length) {
        fprintf(stderr,
                "frost: index %lld out of bounds for length %lld\n",
                (long long)index, (long long)length);
        abort();
    }
}

/* Bounds-check and answer with the index, so an array or slice access can be
   checked inline in an expression: `data[frost_rt_check_index(i, len)]`. */
int64_t frost_rt_check_index(int64_t index, int64_t length) {
    frost_rt_bounds_check(index, length);
    return index;
}

void frost_rt_generation_check(int64_t stored, int64_t expected) {
    if (stored != expected) {
        fprintf(stderr,
                "frost: stale handle, slot generation %lld but handle expected %lld\n",
                (long long)stored, (long long)expected);
        abort();
    }
}

/* Validate a handle against a slab and answer with the slot it names. The low
   32 bits are the index and the high 32 the generation. The index is bounds
   checked and the generation matched against the slot's, so `slab[handle]`
   reading a released or out-of-range slot aborts rather than seeing whatever
   took its place. */
int64_t frost_rt_slot(int64_t handle, int64_t count, const int64_t *generations) {
    int64_t index = handle & 0xffffffff;
    int64_t generation = handle >> 32;
    frost_rt_bounds_check(index, count);
    frost_rt_generation_check(generations[index], generation);
    return index;
}

int64_t frost_rt_byte_at(const char *text, int64_t index) {
    return (int64_t)(unsigned char)text[index];
}

int64_t frost_rt_str_len(const char *text) {
    int64_t length = 0;
    while (text[length] != 0) {
        length++;
    }
    return length;
}

/* Emitted output goes to standard output unless a file has been opened for it,
   which is how `-o` works without every emitter having to carry a destination. */
static FILE *frost_rt_emit_target = 0;

static FILE *frost_rt_emit_where(void) {
    if (frost_rt_emit_target == 0) {
        return stdout;
    }
    return frost_rt_emit_target;
}

int64_t frost_rt_emit_open(const char *path) {
    frost_rt_emit_target = fopen(path, "wb");
    return frost_rt_emit_target != 0;
}

void frost_rt_emit_close(void) {
    if (frost_rt_emit_target != 0) {
        fclose(frost_rt_emit_target);
        frost_rt_emit_target = 0;
    }
}

void frost_rt_emit_str(const char *text) {
    fputs(text, frost_rt_emit_where());
}

/* Emit a counted run of bytes rather than a NUL-terminated string, so the
   caller passes a length-carrying `str` and the read is bounded by it. This is
   what lets the emit path be safe. Nothing scans for a terminator. */
void frost_rt_emit_bytes(const char *data, int64_t length) {
    fwrite(data, 1, (size_t)length, frost_rt_emit_where());
}

void frost_rt_emit_int(int64_t value) {
    fprintf(frost_rt_emit_where(), "%lld", (long long)value);
}

void frost_rt_emit_char(int64_t byte) {
    fputc((int)byte, frost_rt_emit_where());
}

/* The `print` statement. An integer prints as %lld and a float as %g, each
   followed by a newline, and both go to standard output rather than the emit
   target, since `print` is a program writing its own output, not the compiler
   writing its. This matches the self-hosted compiler's `print`, which emits the
   same two printf forms inline. */
void frost_rt_print_i64(int64_t value) {
    printf("%lld\n", (long long)value);
}

void frost_rt_print_f64(double value) {
    printf("%g\n", value);
}

/* The pieces a formatted `print` expands into. Each writes one piece with no
   newline of its own, and the newline is written once at the end, so
   `print "x {} y {}", a, b` is one line however many pieces it took. */
void frost_rt_write_bytes(const char *data, int64_t length) {
    fwrite(data, 1, (size_t)length, stdout);
}

void frost_rt_write_i64(int64_t value) {
    printf("%lld", (long long)value);
}

void frost_rt_write_f64(double value) {
    printf("%g", value);
}

void frost_rt_write_newline(void) {
    printf("\n");
}

/* A C string, whose length is where its NUL is. A string literal reaches C as
   one of these, so this is what writing one comes to when it has no length
   beside it. */
void frost_rt_write_cstr(const char *text) {
    fputs(text, stdout);
}

const char *frost_rt_getenv(const char *name) {
    const char *value = getenv(name);
    if (value == 0) {
        return "";
    }
    return value;
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

void frost_rt_byte_set(char *buffer, int64_t index, int64_t value) {
    buffer[index] = (char)value;
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
    abort();
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
// Reads an i64 through a pointer. A `linear` resource passed to an extern
// arrives as a pointer to the moved-in aggregate (docs/book/src/impl/c-compatibility.md), and
// this is the smallest terminal consumer that proves the value crossed intact.
int64_t frost_rt_read_i64(void *data) {
    return *(int64_t *)data;
}

void frost_rt_error(const char *text) {
    fputs(text, stderr);
}

/* Write a counted run of bytes to stderr, so a diagnostic composed from a `str`
   is bounded by the length it carries rather than a NUL. */
void frost_rt_error_bytes(const char *data, int64_t length) {
    fwrite(data, 1, (size_t)length, stderr);
}

void frost_rt_error_src(const char *text, int64_t offset, int64_t length) {
    fwrite(text + offset, 1, (size_t)length, stderr);
}

void frost_rt_error_int(int64_t value) {
    fprintf(stderr, "%lld", (long long)value);
}

void frost_rt_die(void) {
    fputc('\n', stderr);
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

void *frost_rt_heap_alloc(int64_t size) {
    void *block = malloc((size_t)size);
    if (block != NULL) {
        frost_rt_heap_blocks += 1;
    }
    return block;
}

void *frost_rt_heap_realloc(void *block, int64_t size) {
    void *moved = realloc(block, (size_t)size);
    if (block == NULL && moved != NULL) {
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

/* Copy `size` bytes from `source` to `destination`, for a container growing its
   storage or shifting elements. */
void frost_rt_mem_copy(void *destination, const void *source, int64_t size) {
    memcpy(destination, source, (size_t)size);
}

/* Write `size` bytes of `value` at `destination`, for zeroing a freshly built
   aggregate such as a columns container. */
void frost_rt_mem_set(void *destination, int64_t value, int64_t size) {
    memset(destination, (int)value, (size_t)size);
}

/* Whole-file read and write, for a standard library that does its own IO
   rather than reaching for the C library directly. The read returns a fresh
   heap block the caller frees. The length comes back through frost_rt_file_size. */
static int64_t frost_rt_last_read_length = 0;

const char *frost_rt_file_read(const char *path) {
    FILE *file = fopen(path, "rb");
    if (file == 0) {
        frost_rt_last_read_length = -1;
        return "";
    }
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buffer = (char *)malloc((size_t)length + 1);
    size_t read = fread(buffer, 1, (size_t)length, file);
    buffer[read] = 0;
    fclose(file);
    frost_rt_last_read_length = (int64_t)read;
    return buffer;
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

int64_t frost_rt_remove_file(const char *path) {
    return remove(path) == 0;
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
