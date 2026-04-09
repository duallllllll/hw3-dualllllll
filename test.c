#include "llm_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define TEST(cond, msg) do { if (cond) printf("[PASS] %s\n", msg); else { printf("[FAIL] %s\n", msg); exit(1); } } while(0)

int main() 
{
    const int NUM_BLOCKS = 256;
    size_t pool_size = NUM_BLOCKS * BLOCK_SIZE;
    void *pool = malloc(pool_size);
    assert(pool != NULL);

    LLMEngine *engine = llm_init(pool, pool_size);
    TEST(engine != NULL, "llm_init success");
    TEST(engine->total_local_blocks == NUM_BLOCKS, "total blocks correct");
    TEST(engine->free_local_blocks == NUM_BLOCKS, "free blocks init correct");

    Sequence *seq1 = create_sequence(engine, 1, 10);
    TEST(seq1 != NULL, "create_sequence seq1 success");
    TEST(seq1->seq_id == 1, "seq_id set");
    TEST(seq1->kv_count == 10, "kv_count set");
    TEST(engine->free_local_blocks == NUM_BLOCKS - 1, "one block allocated");

    Sequence *dup = create_sequence(engine, 1, 5);
    TEST(dup == NULL, "create_sequence duplicate id fails");

    Sequence *seq2 = create_sequence(engine, 2, 20);
    TEST(seq2 != NULL, "create_sequence seq2 success");
    TEST(engine->free_local_blocks == NUM_BLOCKS - 3, "3 blocks used total");

    int ret = append_kv(engine, 1, 100);
    TEST(ret == 0, "append_kv to seq1 (within same block)");
    TEST(seq1->kv_count == 11, "kv_count incremented");

    for (int i = 0; i < 5; i++) append_kv(engine, 1, 200 + i);
    TEST(seq1->kv_count == 16, "block 0 full");
    ret = append_kv(engine, 1, 999);
    TEST(ret == 0, "append_kv triggers new block allocation");
    TEST(seq1->kv_count == 17, "kv_count=17");
    TEST(engine->free_local_blocks == NUM_BLOCKS - 4, "new block allocated (4 used)");

    Sequence *seq3 = fork_sequence(engine, 1, 3);
    TEST(seq3 != NULL, "fork_sequence from seq1 to seq3");
    TEST(seq3->kv_count == seq1->kv_count, "child inherits kv_count");

    ret = append_kv(engine, 1, 777);
    TEST(ret == 0, "append to parent triggers CoW");
    TEST(engine->free_local_blocks == NUM_BLOCKS - 5, "CoW allocated one more block");

    ret = free_sequence(engine, 2);
    TEST(ret == 0, "free_sequence seq2");
    TEST(engine->free_local_blocks == NUM_BLOCKS - 3, "blocks freed back");

    ret = free_sequence(engine, 99);
    TEST(ret == -1, "free_sequence invalid id returns -1");

    ret = free_sequence(engine, 1);
    TEST(ret == 0, "free_sequence seq1");
    ret = free_sequence(engine, 3);
    TEST(ret == 0, "free_sequence seq3");
    TEST(engine->free_local_blocks == NUM_BLOCKS, "all blocks freed");

    Sequence *big = create_sequence(engine, 4, MAX_LOGICAL_BLOCKS * KV_PER_BLOCK);
    TEST(big != NULL, "create_sequence at max capacity");
    ret = append_kv(engine, 4, 123);
    TEST(ret == -1, "append beyond max logical blocks fails");

    int remaining = engine->free_local_blocks;
    Sequence *oom_seq = create_sequence(engine, 5, (remaining + 1) * KV_PER_BLOCK);
    TEST(oom_seq == NULL, "create_sequence fails when not enough free blocks");

    ret = llm_destroy(engine);
    TEST(ret == 0, "llm_destroy success");
    free(pool);
    printf("\nAll tests passed!\n");
    return 0;
}