import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, 
                 attn_pdrop=None, residual_pdrop=None, attn_dropout=None, residual_dropout=None):
        super().__init__()
        
        # Support both parameter naming conventions
        attn_pdrop = attn_pdrop if attn_pdrop is not None else attn_dropout
        residual_pdrop = residual_pdrop if residual_pdrop is not None else residual_dropout
        
        if attn_pdrop is None:
            attn_pdrop = 0.0
        if residual_pdrop is None:
            residual_pdrop = 0.0
            
        # Store configuration attributes  
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop)
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model)
        self.dropout = nn.Dropout(residual_pdrop)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x_1 = self.token_embeddings(x)
        position_indices = torch.arange(x.size(1)).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x_2 = self.position_embeddings(position_indices)
        x = x_1 + x_2
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
    
    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=None):
        """Generate text from input_ids"""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Check if we've reached context length limit
                if generated.size(1) >= self.context_length:
                    break
                    
                # Truncate to context length if needed
                current_input = generated
                if current_input.size(1) > self.context_length:
                    current_input = current_input[:, -self.context_length:]
                
                # Forward pass
                logits = self.forward(current_input)
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    values, indices = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < values[:, [-1]]] = -torch.inf
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])