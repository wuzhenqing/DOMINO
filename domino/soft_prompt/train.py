import sys
import logging
import os
import torch
import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    DataCollatorForSeq2Seq
)
from dataclasses import dataclass, field
from typing import Optional
from .model import PromptTuningModel
from .dataset import SoftDataset

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training (uses sdpa when False)."},
    )

@dataclass
class DataArguments:
    train_data_path: str = field(
        metadata={"help": "Path to the json training data."}
    )
    valid_data_path: str = field(
        metadata={"help": "Path to the json valid data."}
    )
    max_seq_len: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum length of training samples."}
    )
    soft_token_count: Optional[int] = field(
        default=None, 
        metadata={"help": "Soft token count"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(os.path.join(training_args.output_dir, "training.log"), mode='a'),]
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, bfloat16: {training_args.bf16}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=model_args.use_fast_tokenizer, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    
    logger.info(f"Tokenizer Class: {tokenizer.__class__.__name__}")
    logger.info(f"PAD Token/Id: {tokenizer.pad_token}/{tokenizer.pad_token_id}")
    logger.info(f"BOS Token/Id: {tokenizer.bos_token}/{tokenizer.bos_token_id}")
    logger.info(f"EOS Token/Id: {tokenizer.eos_token}/{tokenizer.eos_token_id}")

    train_dataset = SoftDataset(data_args.train_data_path, tokenizer, data_args.max_seq_len)
    valid_dataset = SoftDataset(data_args.valid_data_path, tokenizer, data_args.max_seq_len)

    num_samples = len(train_dataset)
    logger.info(f"Train dataset size = {num_samples}")
    logger.info(f"Sample 0: {train_dataset[0]}")
    logger.info(f"Sample 0 input: {tokenizer.decode(train_dataset[0]['input_ids'])}")

    attn_implementation = "flash_attention_2" if model_args.use_flash_attn else "sdpa"
    model = PromptTuningModel(model_args.model_name_or_path, soft_token_count=data_args.soft_token_count, attn_implementation=attn_implementation)
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable() 

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("=" * 80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Trainable%: {100 * trainable_params / total_params:.2f}%")
    logger.info("=" * 80)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    save_soft_token_embeddings(model, training_args.output_dir)

    for obj in trainer.state.log_history:
        logger.info(str(obj))
    
    print("Train Done!")
    

def save_soft_token_embeddings(model, output_dir):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, "soft_token_embeddings.pth")
        torch.save(model.soft_tokens_embeddings.state_dict(), save_path)
        logger.info(f"Soft token embeddings saved to {save_path}")

if __name__ == "__main__":
    main()
