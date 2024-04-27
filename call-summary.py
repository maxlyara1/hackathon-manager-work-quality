import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration 

tokenizer = GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer', eos_token='</s>')
model = T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer')
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check if CUDA is available
model.to(device)

input_text = txt_punctuated

# Encode input text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate outputs
outputs = model.generate(
    input_ids,
    eos_token_id=tokenizer.eos_token_id,
    num_beams=5,
    min_length=50,  # Change 'min_new_tokens' to 'min_length'
    max_length=300,  # Change 'max_new_tokens' to 'max_length'
    do_sample=True,
    no_repeat_ngram_size=4,
    top_p=0.9
)
result = tokenizer.decode(outputs[0][1:]).replace('</s>', '')
print(len(input_text))
print(len(result))
print(result)