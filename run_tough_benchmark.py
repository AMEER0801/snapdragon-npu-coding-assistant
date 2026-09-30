import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

print("⏳ Loading model onto NVIDIA H200 (GPU 0) for stress testing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Model loaded! Ready for the Stress Test Benchmark Suite.\n")

# Top 10 Tough Coding Problems & Dynamic Evaluation Prompts
TOUGH_PROBLEMS = [
    ("1. Algorithmic Optimization", "Write an optimized Python function to find the maximum path sum in a binary tree. The path may start and end at any node. Provide time and space complexity."),
    ("2. Advanced Data Structure", "Implement a complete LRU (Least Recently Used) Cache system in Python using a doubly linked list and a hash map. Include get() and put() methods with O(1) time complexity."),
    ("3. Concurrency & Threading", "Write a thread-safe implementation of a generic Producer-Consumer queue in Python using the threading module and condition variables. Avoid using the built-in queue library."),
    ("4. Dynamic Programming (Edge Case)", "Solve the 'Edit Distance' problem in Python. Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2. Optimize for spatial memory."),
    ("5. System Design Architecture", "Draft a clean Python blueprint class matching a rate-limiter middleware using the Token Bucket algorithm for an API gateway system."),
    ("6. Memory Management", "Write a Python script that detects cycles in a custom directed graph object using deep recursion, and implement an iterative version to completely avoid Python RecursionError crashes on large inputs."),
    ("7. Bit Manipulation", "Write an ultra-fast function in Python that takes an unsigned integer and returns the number of '1' bits it has (Hamming weight), utilizing bitwise operations without typecasting to a string."),
    ("8. SQL / Query Tuning", "Write a complex SQL query to find the top 3 highest-paid employees in each department given an Employee (id, name, salary, departmentId) table and a Department (id, name) table. Optimize for large-scale production databases."),
    ("9. Network & Async IO", "Write a high-performance asynchronous web scraper in Python using asyncio and aiohttp that downloads structural text from 5 URLs simultaneously with explicit error handling and timeout boundaries."),
    ("10. Micro-Optimization / Pythonic Cleanliness", "Refactor a deeply nested block of 5 iterative 'for' loops containing multiple conditional flags into clean, readable, high-performance generator comprehensions without losing operational logic.")
]

print("="*60)
print("       🔥 STARTING THE 10 TOUGH PROBLEMS STRESS TEST 🔥")
print("="*60)

for index, (title, prompt) in enumerate(TOUGH_PROBLEMS, start=1):
    print(f"\n▶️ Running Challenge {title}...")
    
    messages = [
        {"role": "system", "content": "You are a world-class principal software engineer. Provide high-quality, optimal, production-ready code blocks with minimal text explanations."},
        {"role": "user", "content": prompt}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to("cuda:0")
    
    start_time = time.time()
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=400,
            temperature=0.1,  # Low temperature for highly precise code generation
            do_sample=False
        )
        
    end_time = time.time()
    
    # Calculate performance analytics
    generated_tokens_count = len(generated_ids[0]) - len(model_inputs.input_ids[0])
    tokens_per_second = generated_tokens_count / (end_time - start_time)
    
    new_tokens = generated_ids[0][model_inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print("-" * 50)
    print(output_text)
    print("-" * 50)
    print(f"📊 Telemetry Metrics: Generated {generated_tokens_count} tokens in {end_time - start_time:.2f}s ({tokens_per_second:.1f} tokens/sec)")
    print("="*60)

print("\n🎉 All 10 challenges completed! Review the outputs to evaluate your model's accuracy.")