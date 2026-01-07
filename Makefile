ODIN      := odin
GLSLC     := glslc
BIN_DIR   := bin
SHADER_DIR:= shaders

EXE_NAME  := sdlodn
TARGET    := $(BIN_DIR)/$(EXE_NAME)
SRC_DIR   := src

SRC_VERT  := $(SHADER_DIR)/shader.vert
SRC_FRAG  := $(SHADER_DIR)/shader.frag

OUT_VERT  := $(SHADER_DIR)/vert.spv
OUT_FRAG  := $(SHADER_DIR)/frag.spv

ARGS      :=

.PHONY: all build run dev clean shaders

all: build

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(SHADER_DIR):
	@mkdir -p $(SHADER_DIR)

shaders: $(SHADER_DIR) $(OUT_VERT) $(OUT_FRAG)

$(OUT_VERT): $(SRC_VERT)
	@echo "Compiling vertex shader..."
	$(GLSLC) $< -o $@

$(OUT_FRAG): $(SRC_FRAG)
	@echo "Compiling fragment shader..."
	$(GLSLC) $< -o $@

build: shaders $(BIN_DIR)
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
	rm -f $(OUT_VERT) $(OUT_FRAG)
