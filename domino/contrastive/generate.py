import os
import torch
import json 
import argparse
from tqdm import tqdm 
from vllm import SamplingParams, LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_sample_with_transformers(
    pretrained_model_name_or_path,
    tokenizer_name_or_path,
    soft_prompt_dir,
    public_soft_token_count,
    temp,
    target_count,
    max_tokens,
    device,
):
    pre_trained_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, local_files_only=True)
    pre_trained_llm.to(device)
    pre_trained_llm.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, local_files_only=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    public_soft_token_embeddings_path = os.path.join(soft_prompt_dir, 'public_soft_token_embeddings.pth')
    public_soft_token_embeddings_state_dict = torch.load(public_soft_token_embeddings_path, map_location=device)  
    public_soft_tokens_embeddings = torch.nn.Embedding(public_soft_token_count, pre_trained_llm.config.hidden_size)
    public_soft_tokens_embeddings.load_state_dict(public_soft_token_embeddings_state_dict)
    public_soft_tokens_embeddings.to(device).to(pre_trained_llm.dtype).eval()

    soft_token_id = torch.arange(public_soft_token_count).to(device)
    public_soft_token_embeds = public_soft_tokens_embeddings(soft_token_id)
    public_soft_token_embeds = public_soft_token_embeds.view(1, public_soft_token_count, pre_trained_llm.config.hidden_size)
    print(f"soft token embeds shape = {public_soft_token_embeds.shape}")

    soft_attention_mask = torch.ones(1, public_soft_token_count).to(device)

    generated_samples = []
    with torch.no_grad():
        for current in tqdm(range(target_count), desc="Synthetic Samples..."):
            generated_outputs = pre_trained_llm.generate(
                inputs_embeds=public_soft_token_embeds,
                attention_mask=soft_attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temp,
                use_cache=False,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.0
            )

            generated_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

            print("****Generated Text:****")
            print(generated_text)
            print("=="*30)

            generated_samples.append(dict(idx=current, generated_text=generated_text))

    target_path = os.path.join(soft_prompt_dir, f'transformers_generated_{len(generated_samples)}_samples_temp{temp}.jsonl')
    with open(target_path, 'w') as h:
        for generated_sample in generated_samples:
            h.write(json.dumps(generated_sample) + "\n")
    print(f"Saved to {target_path}")


def generate_sample_with_vllm(
    pretrained_model_name_or_path,
    tokenizer_name_or_path,
    soft_prompt_dir,
    public_soft_token_count,
    temp,
    target_count,
    max_tokens,
    batch_size,
    tensor_parallel_size,
    gpu_memory_utilization,
    max_model_len,
    device='cuda:0'
):
    pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True, local_files_only=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    soft_token_embeddings_path = os.path.join(soft_prompt_dir, 'public_soft_token_embeddings.pth')
    soft_token_embeddings_state_dict = torch.load(soft_token_embeddings_path, map_location=device)  
    soft_tokens_embeddings = torch.nn.Embedding(public_soft_token_count, pretrained_model.config.hidden_size)
    soft_tokens_embeddings.load_state_dict(soft_token_embeddings_state_dict)
    soft_tokens_embeddings.to(device).to(pretrained_model.dtype)
    soft_tokens_embeddings.eval()

    new_tokens = [f"<soft_{i}>" for i in range(public_soft_token_count)]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    pretrained_model.resize_token_embeddings(len(tokenizer))
    
    with torch.no_grad():
        pretrained_model.get_input_embeddings().weight[-public_soft_token_count:] = soft_tokens_embeddings.weight.data

    # Prevent soft-token IDs from being emitted as output.
    # Qwen2.5 ties input/output embeddings; materialize a separate lm_head
    # and set the soft-token rows to a large negative logit.
    pretrained_model.config.tie_word_embeddings = False
    new_lm_head = torch.nn.Linear(
        pretrained_model.config.hidden_size,
        len(tokenizer),
        bias=False,
        device=pretrained_model.device,
        dtype=pretrained_model.dtype,
    )
    new_lm_head.weight.data.copy_(pretrained_model.get_input_embeddings().weight.data)
    with torch.no_grad():
        new_lm_head.weight[-public_soft_token_count:] = -1e4
    pretrained_model.lm_head = new_lm_head

    temp_dir = os.path.join(soft_prompt_dir, "vllm_temp_model")
    print("save vllm temp model...")
    # Always overwrite so that embedding-injection fixes take effect.
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    pretrained_model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)
    print("save vllm temp model done!")

    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    del pretrained_model  
    
    llm = LLM(
        model=temp_dir,
        tokenizer=temp_dir,
        dtype='bfloat16',
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temp,
        top_p=1,
        repetition_penalty=1.0,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
    )
    
    prompt = "".join(new_tokens) 
    num_repetitions = (target_count + batch_size - 1) // batch_size
    
    all_synthetic_texts = []
    idx = 0
    for _ in tqdm(range(num_repetitions), desc='Synthetic...'):
        outputs = llm.generate([prompt]*batch_size, sampling_params)
        for output in outputs:
            instruct = output.outputs[0].text.strip()
            all_synthetic_texts.append(dict(idx=idx, synthetic_text=instruct))
            idx += 1
    
    target_path = os.path.join(soft_prompt_dir, f'vllm_generated_{len(all_synthetic_texts)}_samples_temp{temp}.jsonl')
    with open(target_path, 'w') as f:
        for item in all_synthetic_texts:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sythetic samples.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True)
    parser.add_argument("--soft_prompt_dir", type=str, required=True)
    parser.add_argument("--inference_engine", type=str, default="vllm", choices=["transformers", "vllm"])
    parser.add_argument("--public_soft_token_count", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--target_count", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=4608)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.inference_engine == "transformers":
        generate_sample_with_transformers(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            soft_prompt_dir=args.soft_prompt_dir,
            public_soft_token_count=args.public_soft_token_count,
            temp=args.temp,
            target_count=args.target_count,
            max_tokens=args.max_tokens,
            device=args.device
        )
    else:
        generate_sample_with_vllm(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            soft_prompt_dir=args.soft_prompt_dir,
            public_soft_token_count=args.public_soft_token_count,
            temp=args.temp,
            target_count=args.target_count,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            device=args.device
        )
