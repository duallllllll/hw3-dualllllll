CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Werror -O2 -ggdb

all: llm_memory.o

llm_memory.o: llm_memory.c llm_memory.h
	$(CC) $(CFLAGS) -c llm_memory.c -o llm_memory.o

clean:
	rm -f llm_memory.o

.PHONY: all clean
