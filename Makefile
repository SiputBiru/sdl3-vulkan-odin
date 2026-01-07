ODIN      := odin
BIN_DIR   := bin
EXE_NAME  := sdlodn
TARGET    := $(BIN_DIR)/$(EXE_NAME)
SRC_DIR   := src

ARGS      :=

.PHONY: all build run dev clean

all: build

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

build: $(BIN_DIR)
	@echo "Building $(EXE_NAME)..."
	$(ODIN) build $(SRC_DIR)  -out:$(TARGET) -debug -show-timings

run: build
	@echo "Running..."
	@echo "-----------------------"
	./$(TARGET) $(ARGS)

dev: run

clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)
