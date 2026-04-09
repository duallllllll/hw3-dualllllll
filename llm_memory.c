#include "llm_memory.h"
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define container_of(ptr, type, member) ((type *)((char *)(ptr) - offsetof(type, member)))

static Sequence *find_sequence(LLMEngine *engine, int seq_id) {
    struct list_node *cur = engine->active_seqs.next;
    while (cur != &engine->active_seqs) {
        Sequence *seq = container_of(cur, Sequence, node);
        if (seq->seq_id == seq_id)
            return seq;
        cur = cur->next;
    }
    return NULL;
}

static int allocate_physical_block(LLMEngine *engine) {
    for (int i = 0; i < engine->total_local_blocks; i++) {
        if (engine->ref_count[i] == 0)
            return i;
    }
    return -1;
}

LLMEngine *llm_init(void *pool_addr, size_t pool_size) {
    if (pool_addr == NULL || pool_size < BLOCK_SIZE)
        return NULL;
    LLMEngine *engine = (LLMEngine *)malloc(sizeof(LLMEngine));
    if (engine == NULL)
        return NULL;
    engine->pool_base_ptr = pool_addr;
    engine->total_local_blocks = (int)(pool_size / BLOCK_SIZE);
    engine->free_local_blocks = engine->total_local_blocks;
    engine->ref_count = (int *)calloc(engine->total_local_blocks, sizeof(int));
    if (engine->ref_count == NULL) {
        free(engine);
        return NULL;
    }
    engine->active_seqs.prev = &engine->active_seqs;
    engine->active_seqs.next = &engine->active_seqs;
    return engine;
}

Sequence *create_sequence(LLMEngine *engine, int seq_id, int initial_kv_len) {
    if (engine == NULL || initial_kv_len <= 0)
        return NULL;
    if (find_sequence(engine, seq_id) != NULL)
        return NULL;
    int required_blocks = (initial_kv_len + KV_PER_BLOCK - 1) / KV_PER_BLOCK;
    if (required_blocks > MAX_LOGICAL_BLOCKS || required_blocks > engine->free_local_blocks)
        return NULL;
    Sequence *seq = (Sequence *)malloc(sizeof(Sequence));
    if (seq == NULL)
        return NULL;
    seq->seq_id = seq_id;
    seq->kv_count = initial_kv_len;
    seq->block_table = (int **)malloc(L1_PT_SIZE * sizeof(int *));
    if (seq->block_table == NULL) {
        free(seq);
        return NULL;
    }
    for (int i = 0; i < L1_PT_SIZE; i++)
        seq->block_table[i] = NULL;
    int allocated_blocks = 0;
    int success = 1;
    for (int logical_block = 0; logical_block < required_blocks; logical_block++) {
        int l1_idx = logical_block / L2_PT_SIZE;
        int l2_idx = logical_block % L2_PT_SIZE;
        if (seq->block_table[l1_idx] == NULL) {
            int *l2 = (int *)malloc(L2_PT_SIZE * sizeof(int));
            if (l2 == NULL) {
                success = 0;
                break;
            }
            for (int j = 0; j < L2_PT_SIZE; j++)
                l2[j] = -1;
            seq->block_table[l1_idx] = l2;
        }
        int block_id = allocate_physical_block(engine);
        if (block_id == -1) {
            success = 0;
            break;
        }
        seq->block_table[l1_idx][l2_idx] = block_id;
        engine->ref_count[block_id] = 1;
        engine->free_local_blocks--;
        allocated_blocks++;
    }
    if (!success) {
        for (int i = 0; i < L1_PT_SIZE; i++) {
            if (seq->block_table[i] != NULL) {
                int *l2 = seq->block_table[i];
                for (int j = 0; j < L2_PT_SIZE; j++) {
                    int blk = l2[j];
                    if (blk != -1) {
                        engine->ref_count[blk]--;
                        if (engine->ref_count[blk] == 0)
                            engine->free_local_blocks++;
                    }
                }
                free(l2);
            }
        }
        free(seq->block_table);
        free(seq);
        return NULL;
    }
    seq->node.prev = engine->active_seqs.prev;
    seq->node.next = &engine->active_seqs;
    engine->active_seqs.prev->next = &seq->node;
    engine->active_seqs.prev = &seq->node;
    return seq;
}

int free_sequence(LLMEngine *engine, int seq_id) {
    if (engine == NULL)
        return -1;
    Sequence *seq = find_sequence(engine, seq_id);
    if (seq == NULL)
        return -1;
    seq->node.prev->next = seq->node.next;
    seq->node.next->prev = seq->node.prev;
    for (int i = 0; i < L1_PT_SIZE; i++) {
        if (seq->block_table[i] != NULL) {
            int *l2 = seq->block_table[i];
            for (int j = 0; j < L2_PT_SIZE; j++) {
                int blk = l2[j];
                if (blk != -1) {
                    engine->ref_count[blk]--;
                    if (engine->ref_count[blk] == 0)
                        engine->free_local_blocks++;
                }
            }
            free(l2);
        }
    }
    free(seq->block_table);
    free(seq);
    return 0;
}

Sequence *fork_sequence(LLMEngine *engine, int parent_seq_id, int new_seq_id) {
    if (engine == NULL)
        return NULL;
    Sequence *parent = find_sequence(engine, parent_seq_id);
    if (parent == NULL)
        return NULL;
    if (find_sequence(engine, new_seq_id) != NULL)
        return NULL;
    Sequence *child = (Sequence *)malloc(sizeof(Sequence));
    if (child == NULL)
        return NULL;
    child->seq_id = new_seq_id;
    child->kv_count = parent->kv_count;
    child->block_table = (int **)malloc(L1_PT_SIZE * sizeof(int *));
    if (child->block_table == NULL) {
        free(child);
        return NULL;
    }
    for (int i = 0; i < L1_PT_SIZE; i++)
        child->block_table[i] = NULL;
    int success = 1;
    for (int i = 0; i < L1_PT_SIZE; i++) {
        if (parent->block_table[i] != NULL) {
            int *l2 = (int *)malloc(L2_PT_SIZE * sizeof(int));
            if (l2 == NULL) {
                success = 0;
                break;
            }
            for (int j = 0; j < L2_PT_SIZE; j++)
                l2[j] = -1;
            child->block_table[i] = l2;
            for (int j = 0; j < L2_PT_SIZE; j++) {
                int blk = parent->block_table[i][j];
                if (blk != -1) {
                    l2[j] = blk;
                    engine->ref_count[blk]++;
                }
            }
        }
    }
    if (!success) {
        for (int i = 0; i < L1_PT_SIZE; i++) {
            if (child->block_table[i] != NULL) {
                int *l2 = child->block_table[i];
                for (int j = 0; j < L2_PT_SIZE; j++) {
                    int blk = l2[j];
                    if (blk != -1) {
                        engine->ref_count[blk]--;
                        if (engine->ref_count[blk] == 0)
                            engine->free_local_blocks++;
                    }
                }
                free(l2);
            }
        }
        free(child->block_table);
        free(child);
        return NULL;
    }
    child->node.prev = engine->active_seqs.prev;
    child->node.next = &engine->active_seqs;
    engine->active_seqs.prev->next = &child->node;
    engine->active_seqs.prev = &child->node;
    return child;
}

int append_kv(LLMEngine *engine, int seq_id, int kv_data) {
    if (engine == NULL)
        return -1;
    Sequence *seq = find_sequence(engine, seq_id);
    if (seq == NULL)
        return -1;
    int logical_idx = seq->kv_count / KV_PER_BLOCK;
    int offset = seq->kv_count % KV_PER_BLOCK;
    if (logical_idx >= MAX_LOGICAL_BLOCKS)
        return -1;
    int l1_idx = logical_idx / L2_PT_SIZE;
    int l2_idx = logical_idx % L2_PT_SIZE;
    if (seq->block_table[l1_idx] == NULL) {
        int *l2 = (int *)malloc(L2_PT_SIZE * sizeof(int));
        if (l2 == NULL)
            return -1;
        for (int i = 0; i < L2_PT_SIZE; i++)
            l2[i] = -1;
        seq->block_table[l1_idx] = l2;
    }
    int *l2_entry = &seq->block_table[l1_idx][l2_idx];
    int need_new = 0;
    int cow = 0;
    int old_block = -1;
    if (*l2_entry == -1) {
        need_new = 1;
    } else if (engine->ref_count[*l2_entry] > 1) {
        need_new = 1;
        cow = 1;
        old_block = *l2_entry;
    }
    if (need_new) {
        if (engine->free_local_blocks == 0)
            return -1;
        int new_block = allocate_physical_block(engine);
        if (new_block == -1)
            return -1;
        if (cow) {
            void *src = (char *)engine->pool_base_ptr + old_block * BLOCK_SIZE;
            void *dst = (char *)engine->pool_base_ptr + new_block * BLOCK_SIZE;
            memcpy(dst, src, BLOCK_SIZE);
            engine->ref_count[old_block]--;
            if (engine->ref_count[old_block] == 0)
                engine->free_local_blocks++;
        }
        *l2_entry = new_block;
        engine->ref_count[new_block] = 1;
        engine->free_local_blocks--;
    }
    int block_id = *l2_entry;
    void *block_addr = (char *)engine->pool_base_ptr + block_id * BLOCK_SIZE;
    int *kv_slot = (int *)block_addr + offset;
    *kv_slot = kv_data;
    seq->kv_count++;
    return 0;
}

int llm_destroy(LLMEngine *engine) {
    if (engine == NULL)
        return -1;
    struct list_node *cur = engine->active_seqs.next;
    while (cur != &engine->active_seqs) {
        struct list_node *next = cur->next;
        Sequence *seq = container_of(cur, Sequence, node);
        free_sequence(engine, seq->seq_id);
        cur = next;
    }
    free(engine->ref_count);
    free(engine);
    return 0;
}