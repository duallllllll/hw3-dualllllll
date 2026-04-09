#ifndef LLM_MEMORY_H
#define LLM_MEMORY_H

#include <stddef.h>
#include <stdint.h>

/* --- Global physical/block-table parameters --- */
#define BLOCK_SIZE 64               /* Physical block size: 64 bytes */
#define KV_SIZE 4                   /* Size of one KV record: 4 bytes (int) */
#define KV_PER_BLOCK 16             /* Number of KV records per block */
#define L1_PT_SIZE 16               /* Number of entries in L1 block table */
#define L2_PT_SIZE 16               /* Number of entries in L2 block table */
#define MAX_LOGICAL_BLOCKS 256      /* L1_PT_SIZE * L2_PT_SIZE */

/* --- Intrusive linked-list node --- */
struct list_node {
	struct list_node *prev;
	struct list_node *next;
};

/* --- Sequence control block --- */
typedef struct Sequence {
	int seq_id;
	int **block_table;             /* Points to L1 block table (array of int* to L2 tables) */
	int kv_count;
	struct list_node node;         /* Must stay at the end of the struct */
} Sequence;

/* --- Global LLM engine controller --- */
typedef struct {
	void *pool_base_ptr;           /* Injected externally; do not clear/overwrite in llm_init */
	int total_local_blocks;
	int free_local_blocks;         /* Remaining free physical blocks in local pool */
	int *ref_count;
	struct list_node active_seqs;  /* Head (dummy/sentinel) of intrusive doubly-circular list */
} LLMEngine;

/* API */
LLMEngine *llm_init(void *pool_addr, size_t pool_size);
Sequence *create_sequence(LLMEngine *engine, int seq_id, int initial_kv_len);
int append_kv(LLMEngine *engine, int seq_id, int kv_data);
Sequence *fork_sequence(LLMEngine *engine, int parent_seq_id, int new_seq_id);
int free_sequence(LLMEngine *engine, int seq_id);
int llm_destroy(LLMEngine *engine);

#endif /* LLM_MEMORY_H */
