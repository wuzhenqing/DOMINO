import torch 
from transformers import AutoModelForCausalLM


class PromptTuningModel(torch.nn.Module):
    """
    Shared soft tokens across all samples.
    Task-level prompt tuning.
    """
    def __init__(self, pretrained_model_name, soft_token_count, attn_implementation="sdpa"):
        super().__init__()
        self.pretrained_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name,
        attn_implementation=attn_implementation, use_cache=False, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        self.config = self.pretrained_llm.config

        for param in self.pretrained_llm.parameters():
            param.requires_grad = False

        self.soft_token_count = soft_token_count
        self.hidden_size = self.pretrained_llm.config.hidden_size

        self.soft_tokens_embeddings = torch.nn.Embedding(soft_token_count, self.hidden_size)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs:
            self.pretrained_llm.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
        else:
            self.pretrained_llm.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.size(0)

        inputs_embeds = self.pretrained_llm.get_input_embeddings()(input_ids)

        prompt_indices = torch.arange(self.soft_token_count, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        prompt_embeddings = self.soft_tokens_embeddings(prompt_indices)

        inputs_embeds = torch.cat([prompt_embeddings, inputs_embeds], dim=1)

        soft_attention_mask = torch.ones(batch_size, self.soft_token_count).to(attention_mask.device)
        attention_mask = torch.cat([soft_attention_mask, attention_mask], dim=1)

        labels = torch.cat([torch.full((batch_size, self.soft_token_count), -100).to(input_ids.device), input_ids], dim=1)

        outputs = self.pretrained_llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask,  
            labels=labels)

        return outputs
