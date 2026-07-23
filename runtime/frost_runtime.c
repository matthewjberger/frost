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

void frost_emit_str(const char *text) {
    fputs(text, stdout);
}

void frost_emit_int(int64_t value) {
    printf("%lld", (long long)value);
}

void frost_emit_char(int64_t byte) {
    putchar((int)byte);
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

void frost_test_start(const char *name) {
    printf("test %s ... ", name);
    fflush(stdout);
}

void frost_test_ok(void) {
    printf("ok\n");
    fflush(stdout);
}

void frost_assert(int8_t condition) {
    if (!condition) {
        printf("FAILED\n");
        fflush(stdout);
        fprintf(stderr, "frost: assertion failed\n");
        abort();
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

/* Which calling convention the native backend must emit for. */
int64_t frost_is_windows(void) {
#ifdef _WIN32
    return 1;
#else
    return 0;
#endif
}
