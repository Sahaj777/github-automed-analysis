from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'gpt2-medium'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Save the model and tokenizer to disk
model.save_pretrained('model/')
tokenizer.save_pretrained('tokenizer/')