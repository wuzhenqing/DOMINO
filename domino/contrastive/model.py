import torch 
from transformers import AutoModelForCausalLM
import os 


class PublicPrivateContrastiveModel(torch.nn.Module):
    """
    domain & sample level soft token under contrastive learning.
    """
    def __init__(self, pretrained_model_name, public_soft_token_count, private_soft_token_count, domain_samples, attn_implementation="sdpa"):
        super().__init__()
        self.pretrained_llm = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            local_files_only=True,
        )

        for param in self.pretrained_llm.parameters():
            param.requires_grad = False

        self.public_soft_token_count = public_soft_token_count
        self.private_soft_token_count = private_soft_token_count
        self.domain_samples = domain_samples
        self.hidden_size = self.pretrained_llm.config.hidden_size

        # Domain level soft tokens
        self.public_soft_tokens_embeddings = torch.nn.Embedding(self.public_soft_token_count, self.hidden_size)

        # Sample level soft tokens
        self.private_soft_tokens_embeddings = torch.nn.Embedding(self.domain_samples, self.private_soft_token_count * self.hidden_size)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None, **kwargs):
        """
        gradient checkpointing
        """
        merged_kwargs = dict(gradient_checkpointing_kwargs or {})
        merged_kwargs.update(kwargs)
        if merged_kwargs:
            self.pretrained_llm.gradient_checkpointing_enable(**merged_kwargs)
        else:
            self.pretrained_llm.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, sample_idx):
        batch_size = input_ids.size(0)

        inputs_embeds = self.pretrained_llm.get_input_embeddings()(input_ids)

        public_soft_token_indices = torch.arange(self.public_soft_token_count, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        public_soft_token_embeds = self.public_soft_tokens_embeddings(public_soft_token_indices)

        private_soft_token_embeds = self.private_soft_tokens_embeddings(sample_idx)
        private_soft_token_embeds = private_soft_token_embeds.view(batch_size, self.private_soft_token_count, self.hidden_size)

        loss_public_only, loss_public_private, denominator_matrix = self._compute_losses(
            inputs_embeds,
            attention_mask, 
            public_soft_token_embeds, 
            private_soft_token_embeds, 
            input_ids,
            sample_idx)

        # Compute contrastive loss
        contrastive_loss = self._compute_contrastive_loss(loss_public_only, loss_public_private, denominator_matrix)

        return {
            "loss": contrastive_loss,
            "loss_public_only": loss_public_only.mean(),
            "loss_public_private": loss_public_private.mean(),
        }

    def _compute_losses(self, inputs_embeds, attention_mask, public_soft_token_embeds, private_soft_token_embeds, input_ids, sample_idx):
        batch_size = inputs_embeds.size(0)

        # --- Compute CE loss with only public soft tokens ---
        inputs_embeds_public = torch.cat([public_soft_token_embeds, inputs_embeds], dim=1)
        attention_mask_public = torch.cat([torch.ones(batch_size, self.public_soft_token_count, device=inputs_embeds.device), attention_mask], dim=1)
        labels_public = torch.cat([torch.full((batch_size, self.public_soft_token_count), -100, device=input_ids.device), input_ids], dim=1)

        outputs_public = self.pretrained_llm(
            inputs_embeds=inputs_embeds_public,
            attention_mask=attention_mask_public,
            labels=labels_public
        )
        loss_public_only = outputs_public.loss
        del outputs_public

        # --- Compute CE loss with public + private soft tokens ---
        inputs_embeds_public_private = torch.cat([public_soft_token_embeds, private_soft_token_embeds, inputs_embeds], dim=1)
        attention_mask_public_private = torch.cat([torch.ones(batch_size, self.public_soft_token_count + self.private_soft_token_count, device=inputs_embeds.device), attention_mask], dim=1)
        labels_public_private = torch.cat([torch.full((batch_size, self.public_soft_token_count + self.private_soft_token_count), -100, device=input_ids.device), input_ids], dim=1)

        outputs_public_private = self.pretrained_llm(
            inputs_embeds=inputs_embeds_public_private,
            attention_mask=attention_mask_public_private,
            labels=labels_public_private
        )
        loss_public_private = outputs_public_private.loss
        del outputs_public_private

        # --- Compute denominator (sum of CE losses for all other samples) ---
        denominator_matrix = torch.zeros((batch_size, batch_size), device=inputs_embeds.device)
        for i in range(batch_size):
            current_private_embeds = self.private_soft_tokens_embeddings(torch.full((batch_size,), sample_idx[i], device=inputs_embeds.device))
            current_private_embeds = current_private_embeds.view(batch_size, self.private_soft_token_count, self.hidden_size)
            
            inputs_embeds_denominator = torch.cat([public_soft_token_embeds, current_private_embeds, inputs_embeds], dim=1)
            attention_mask_denominator = torch.cat([torch.ones(batch_size, self.public_soft_token_count + self.private_soft_token_count, device=inputs_embeds.device), attention_mask], dim=1)
            labels_denominator = torch.cat([torch.full((batch_size, self.public_soft_token_count + self.private_soft_token_count), -100, device=input_ids.device), input_ids], dim=1)

            outputs_denominator = self.pretrained_llm(
                inputs_embeds=inputs_embeds_denominator,
                attention_mask=attention_mask_denominator,
                labels=labels_denominator
            )

            shift_logits_denominator = outputs_denominator.logits[..., :-1, :].contiguous()
            shift_labels_denominator = labels_denominator[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            loss_denominator_per_sample = self._compute_per_sample_loss(shift_logits_denominator, shift_labels_denominator, loss_fct, batch_size)

            denominator_matrix[i, :] = loss_denominator_per_sample

        return loss_public_only, loss_public_private, denominator_matrix

    def _compute_contrastive_loss(self, loss_public_only, loss_public_private, denominator_matrix):
        numerator = loss_public_only + loss_public_private
        exp_denominator_matrix = torch.exp(-denominator_matrix)
        matrix_no_diag = exp_denominator_matrix - torch.diag(torch.diag(exp_denominator_matrix))
        denominator = matrix_no_diag.sum() / (denominator_matrix.numel() - denominator_matrix.size(0))
        contrastive_loss = numerator + torch.log(denominator)
        return contrastive_loss

    def _compute_per_sample_loss(self, shift_logits, shift_labels, loss_fct, batch_size):
        """Compute per-sample loss"""
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size, -1)
        
        valid_tokens = (shift_labels != -100).float()
        return (loss_per_token * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
