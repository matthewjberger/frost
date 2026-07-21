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
