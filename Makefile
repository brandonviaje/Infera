# Compiler and Flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Isrc
LDFLAGS = -lprotobuf  

# Directories
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Source Files
PROTO_SRC = $(SRC_DIR)/onnx-ml.pb.cc
GRAPH_SRC = $(SRC_DIR)/graph.cpp

# Targets
TENSOR_TEST_EXE = $(BUILD_DIR)/run_tensor_tests
NODE_TEST_EXE = $(BUILD_DIR)/run_node_tests
GRAPH_TEST_EXE = $(BUILD_DIR)/run_graph_tests

all: test

# Generate Protobuf files 
$(SRC_DIR)/onnx-ml.pb.cc: proto/onnx-ml.proto
	@echo "Generating Protobuf files..."
	protoc --proto_path=proto --cpp_out=$(SRC_DIR) $<

# Run all tests
test: $(TENSOR_TEST_EXE) $(NODE_TEST_EXE) $(GRAPH_TEST_EXE) 
	@echo "--- Running Tensor Tests ---"
	@./$(TENSOR_TEST_EXE)
	@echo "\n--- Running Node Tests ---"
	@./$(NODE_TEST_EXE)
	@echo "\n--- Running Graph Tests ---"
	@./$(GRAPH_TEST_EXE)

# Compile Tensor Tests
$(TENSOR_TEST_EXE): $(TEST_DIR)/tensor_test.cpp
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) $< -o $@

# Compile Node Tests
$(NODE_TEST_EXE): $(TEST_DIR)/node_test.cpp $(PROTO_SRC)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Compile Graph Tests
$(GRAPH_TEST_EXE): $(TEST_DIR)/graph_test.cpp $(PROTO_SRC) $(GRAPH_SRC)
	@mkdir -p $(BUILD_DIR)
	@$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Clean
clean:
	@rm -rf $(BUILD_DIR)
	@echo "Cleaned build directory."

.PHONY: all test clean
