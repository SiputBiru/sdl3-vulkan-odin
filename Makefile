ODIN      := odin
GLSLC     := glslc
BIN_DIR   := bin
EXE_NAME  := sdlodn
TARGET    := $(BIN_DIR)/$(EXE_NAME)
SRC_DIR   := src

SHADER_DIR := shaders
VERT_SRCS  := $(wildcard $(SHADER_DIR)/*.vert)
FRAG_SRCS  := $(wildcard $(SHADER_DIR)/*.frag)

VERT_SPV   := $(patsubst $(SHADER_DIR)/%, $(BIN_DIR)/%.spv, $(VERT_SRCS))
FRAG_SPV   := $(patsubst $(SHADER_DIR)/%, $(BIN_DIR)/%.spv, $(FRAG_SRCS))

ARGS      :=

.PHONY: all build run dev clean shaders

all: build

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

shaders: $(BIN_DIR) $(VERT_SPV) $(FRAG_SPV)

$(BIN_DIR)/%.vert.spv: $(SHADER_DIR)/%.vert
	@echo "Compiling vertex shader: $<"
	$(GLSLC) $< -o $@

$(BIN_DIR)/%.frag.spv: $(SHADER_DIR)/%.frag
	@echo "Compiling fragment shader: $<"
	$(GLSLC) $< -o $@

build: shaders
	@echo "Building $(EXE_NAME)..."
	$(ODIN) build $(SRC_DIR) -out:$(TARGET) -debug -show-timings

run: build
	@echo "Running..."
	@echo "-----------------------"
	./$(TARGET) $(ARGS)

dev: run

clean:
	@echo "Cleaning up..."
	rm -rf $(BIN_DIR)
