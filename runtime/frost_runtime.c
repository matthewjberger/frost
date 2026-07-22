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

typedef struct {
    unsigned char *storage;
    uint32_t *generations;
    uint32_t *free_list;
    int64_t free_count;
    int64_t capacity;
    int64_t elem_size;
} FrostPool;

void *pool_new(int64_t capacity, int64_t elem_size) {
    FrostPool *pool = (FrostPool *)malloc(sizeof(FrostPool));
    pool->storage =
        (unsigned char *)malloc((size_t)(capacity * elem_size));
    pool->generations =
        (uint32_t *)calloc((size_t)capacity, sizeof(uint32_t));
    pool->free_list =
        (uint32_t *)malloc((size_t)capacity * sizeof(uint32_t));
    pool->free_count = capacity;
    pool->capacity = capacity;
    pool->elem_size = elem_size;
    for (int64_t k = 0; k < capacity; k++) {
        pool->free_list[k] = (uint32_t)(capacity - 1 - k);
    }
    return pool;
}

int64_t pool_alloc(void *pool_ptr, void *value) {
    FrostPool *pool = (FrostPool *)pool_ptr;
    if (pool->free_count == 0) {
        return -1;
    }
    pool->free_count--;
    uint32_t index = pool->free_list[pool->free_count];
    memcpy(pool->storage + (size_t)index * (size_t)pool->elem_size, value,
           (size_t)pool->elem_size);
    uint32_t generation = pool->generations[index];
    return ((int64_t)generation << 32) | (int64_t)index;
}

void *pool_get(void *pool_ptr, int64_t handle) {
    FrostPool *pool = (FrostPool *)pool_ptr;
    uint32_t index = (uint32_t)(handle & 0xFFFFFFFF);
    uint32_t generation = (uint32_t)((uint64_t)handle >> 32);
    if ((int64_t)index >= pool->capacity) {
        return 0;
    }
    if (pool->generations[index] != generation) {
        return 0;
    }
    return pool->storage + (size_t)index * (size_t)pool->elem_size;
}

int64_t pool_free(void *pool_ptr, int64_t handle) {
    FrostPool *pool = (FrostPool *)pool_ptr;
    uint32_t index = (uint32_t)(handle & 0xFFFFFFFF);
    uint32_t generation = (uint32_t)((uint64_t)handle >> 32);
    if ((int64_t)index >= pool->capacity) {
        return 0;
    }
    if (pool->generations[index] != generation) {
        return 0;
    }
    pool->generations[index] += 1;
    pool->free_list[pool->free_count] = index;
    pool->free_count++;
    return 1;
}

int64_t pool_contains(void *pool_ptr, int64_t handle) {
    FrostPool *pool = (FrostPool *)pool_ptr;
    uint32_t index = (uint32_t)(handle & 0xFFFFFFFF);
    uint32_t generation = (uint32_t)((uint64_t)handle >> 32);
    if ((int64_t)index >= pool->capacity) {
        return 0;
    }
    return pool->generations[index] == generation ? 1 : 0;
}

int64_t frost_read_i64(void *data) {
    return *(int64_t *)data;
}

int64_t handle_index(int64_t handle) {
    return handle & 0xFFFFFFFF;
}

int64_t handle_generation(int64_t handle) {
    return (int64_t)((uint64_t)handle >> 32);
}

void pool_destroy(void *pool_ptr) {
    FrostPool *pool = (FrostPool *)pool_ptr;
    free(pool->storage);
    free(pool->generations);
    free(pool->free_list);
    free(pool);
}

/* Diagnostics for a Frost-written compiler. Its program output goes to stdout,
   so errors are composed piecewise on stderr and frost_die ends the process. */
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
