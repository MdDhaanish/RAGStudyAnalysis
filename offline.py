from llama_cpp import Llama

# Load your model

llm = Llama.from_pretrained(
	repo_id="TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF",
	filename="capybarahermes-2.5-mistral-7b.Q2_K.gguf",
)


# Generate a response
user_prompt = "Explain the process of photosynthesis in simple terms."
prompt = f"[INST] {user_prompt} [/INST]"

output = llm(prompt, max_tokens=300, stop=["</s>"])
print(output["choices"][0]["text"])
