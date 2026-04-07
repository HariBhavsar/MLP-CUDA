# Variables
NVCC        := nvcc
ARCH        := sm_89
CFLAGS      := -O3
LIBS        := -lcublas
BIN_DIR     := bins

# Define targets explicitly based on source files
SRCS        := partA.cu partB.cu
TARGETS     := $(SRCS:%.cu=$(BIN_DIR)/%.bin)

# Default target
all: $(TARGETS)

# Pattern Rule: This handles partA.bin from partA.cu AND partB.bin from partB.cu
# % is a wildcard that matches the filename stem
$(BIN_DIR)/%.bin: %.cu | $(BIN_DIR)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) $< $(LIBS) -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

.PHONY: all clean partA partB

# Shortcuts to build specific files
partA: $(BIN_DIR)/partA.bin
partB: $(BIN_DIR)/partB.bin

clean:
	rm -rf $(BIN_DIR)