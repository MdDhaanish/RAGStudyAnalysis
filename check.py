import llama_cpp

# Initialize the backend
llama_cpp.llama_backend_init()

# Get system info
print("\n=== System Info ===")
print("GPU Offload Supported:", llama_cpp.llama_supports_gpu_offload())
print("GPU Support:", llama_cpp.llama_supports_gpu_offload())  # True/False

# Create a temporary model to check layers
params = llama_cpp.llama_context_default_params()
params.n_gpu_layers = 999  # Test with unrealistically high number
try:
    model = llama_cpp.llama_init_from_file(b"./models/capybarahermes-2.5-mistral-7b.Q4_0.gguf", params)
    print("\n=== Maximum Layers ===")
    print("Actual GPU Layers Loaded:", params.n_gpu_layers)  # Will show what was actually possible
except Exception as e:
    print(f"Error: {str(e)}")
finally:
    llama_cpp.llama_backend_free()