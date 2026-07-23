#include <setjmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void frost_bounds_check(int64_t index, int64_t length) {
    if ((uint64_t)index >= (uint64_t)length) {
        fprintf(stderr,
                "frost: index %lld out of bounds for length %lld\n",
                (long long)index, (long long)length);
        abort();
    }
}

void frost_generation_check(int64_t stored, int64_t expected) {
    if (stored != expected) {
        fprintf(stderr,
                "frost: stale handle, slot generation %lld but handle expected %lld\n",
                (long long)stored, (long long)expected);
        abort();
    }
}

int64_t frost_byte_at(const char *text, int64_t index) {
    return (int64_t)(unsigned char)text[index];
}

int64_t frost_str_len(const char *text) {
    int64_t length = 0;
    while (text[length] != 0) {
        length++;
    }
    return length;
}

/* Emitted output goes to standard output unless a file has been opened for it,
   which is how `-o` works without every emitter having to carry a destination. */
static FILE *frost_emit_target = 0;

static FILE *frost_emit_where(void) {
    if (frost_emit_target == 0) {
        return stdout;
    }
    return frost_emit_target;
}

int64_t frost_emit_open(const char *path) {
    frost_emit_target = fopen(path, "wb");
    return frost_emit_target != 0;
}

void frost_emit_close(void) {
    if (frost_emit_target != 0) {
        fclose(frost_emit_target);
        frost_emit_target = 0;
    }
}

void frost_emit_str(const char *text) {
    fputs(text, frost_emit_where());
}

void frost_emit_int(int64_t value) {
    fprintf(frost_emit_where(), "%lld", (long long)value);
}

void frost_emit_char(int64_t byte) {
    fputc((int)byte, frost_emit_where());
}

const char *frost_getenv(const char *name) {
    const char *value = getenv(name);
    if (value == 0) {
        return "";
    }
    return value;
}

const char *frost_read_file(const char *path) {
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

void frost_byte_set(char *buffer, int64_t index, int64_t value) {
    buffer[index] = (char)value;
}

/* The test runner. A failing assertion has to end the test it is in without
   ending the run, or one bad test hides every test after it. The escape is a
   longjmp back into frost_test_run, which is why the runner takes the test body
   as a function pointer rather than being a sequence the compiler emits: the
   setjmp has to own the call. */
/* On Win64 longjmp unwinds through SEH, which needs unwind information for
   every frame it passes. The assembly backend emits none, so a test body that
   fails an assertion would fault on the way out rather than escaping. Setting
   the jump with no frame makes longjmp a plain register restore, which is all
   that escaping a hand-written frame needs. */
#if defined(_WIN32) && defined(__GNUC__)
#define frost_setjmp(env) _setjmp((env), 0)
#else
#define frost_setjmp(env) setjmp(env)
#endif

static jmp_buf frost_test_escape;
static int frost_inside_test = 0;
static int64_t frost_tests_passed = 0;
static int64_t frost_tests_failed = 0;

void frost_test_run(const char *name, void (*body)(void)) {
    printf("test %s ... ", name);
    fflush(stdout);
    frost_inside_test = 1;
    if (frost_setjmp(frost_test_escape) == 0) {
        body();
        frost_tests_passed++;
        printf("ok\n");
    } else {
        frost_tests_failed++;
    }
    frost_inside_test = 0;
    fflush(stdout);
}

/* Returns the failure count, so the process can exit non-zero on it. */
int64_t frost_test_summary(void) {
    printf("\n%lld passed, %lld failed\n", (long long)frost_tests_passed,
           (long long)frost_tests_failed);
    fflush(stdout);
    return frost_tests_failed;
}

/* An assertion outside a test has nowhere to escape to, so it still aborts.
   Inside one it fails that test and the run carries on. */
static void frost_assert_failed(const char *where) {
    printf("FAILED\n");
    fflush(stdout);
    if (where != 0) {
        fprintf(stderr, "  assertion failed at %s\n", where);
    } else {
        fprintf(stderr, "  assertion failed\n");
    }
    fflush(stderr);
    if (frost_inside_test) {
        longjmp(frost_test_escape, 1);
    }
    abort();
}

void frost_assert(int8_t condition) {
    if (!condition) {
        frost_assert_failed(0);
    }
}

/* The same assertion carrying the source position the compiler knew, so a
   failure names the line the reader wrote rather than only the test it was in. */
void frost_assert_at(int8_t condition, const char *where) {
    if (!condition) {
        frost_assert_failed(where);
    }
}


/* Diagnostics for a Frost-written compiler. Its program output goes to stdout,
   so errors are composed piecewise on stderr and frost_die ends the process. */
// Reads an i64 through a pointer. A `linear` resource passed to an extern
// arrives as a pointer to the moved-in aggregate (docs/c-compatibility.md), and
// this is the smallest terminal consumer that proves the value crossed intact.
int64_t frost_read_i64(void *data) {
    return *(int64_t *)data;
}

void frost_error(const char *text) {
    fputs(text, stderr);
}

void frost_error_src(const char *text, int64_t offset, int64_t length) {
    fwrite(text + offset, 1, (size_t)length, stderr);
}

void frost_error_int(int64_t value) {
    fprintf(stderr, "%lld", (long long)value);
}

void frost_die(void) {
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
static int frost_argument_count(void) {
    return __argc;
}
static char **frost_argument_vector(void) {
    return __argv;
}
#else
static int frost_saved_argc = 0;
static char **frost_saved_argv = 0;

__attribute__((constructor)) static void frost_capture_arguments(int argc,
                                                                char **argv) {
    frost_saved_argc = argc;
    frost_saved_argv = argv;
}

static int frost_argument_count(void) {
    return frost_saved_argc;
}
static char **frost_argument_vector(void) {
    return frost_saved_argv;
}
#endif

int64_t frost_arg_count(void) {
    return (int64_t)frost_argument_count();
}

/* Out of range answers with the empty string rather than failing, so a caller
   reads arguments by asking rather than by counting first. */
const char *frost_arg_at(int64_t index) {
    if (index < 0 || index >= (int64_t)frost_argument_count()) {
        return "";
    }
    return frost_argument_vector()[index];
}

int64_t frost_remove_file(const char *path) {
    return remove(path) == 0;
}

/* Runs a command line through the shell and answers with its exit status, so
   the compiler can drive the assembler and the linker. */
int64_t frost_run_command(const char *command) {
    int status = system(command);
    return (int64_t)status;
}

/* Which calling convention the native backend must emit for. */
int64_t frost_is_windows(void) {
#ifdef _WIN32
    return 1;
#else
    return 0;
#endif
}
