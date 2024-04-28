import torch
from transformers import GPT2Tokenizer, T5ForConditionalGeneration 
from sbert_punc_case_ru import SbertPuncCase

class ConversationSummarizer:
    sbert_punc_case_model = None
    tokenizer = None
    model = None
    device = None

    def __init__(self):
        if not ConversationSummarizer.sbert_punc_case_model:
            ConversationSummarizer.sbert_punc_case_model = SbertPuncCase()
        if not ConversationSummarizer.tokenizer:
            ConversationSummarizer.tokenizer = GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer', eos_token='</s>')
        if not ConversationSummarizer.model:
            ConversationSummarizer.model = T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer')
        if not ConversationSummarizer.device:
            ConversationSummarizer.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ConversationSummarizer.model.to(ConversationSummarizer.device)

    def add_puctuation_in_text(self, conversation_df, conversation_id):
        input_text = ' '.join(conversation_df[conversation_df['conversation_id'] == conversation_id].apply(lambda row: ': '.join([row['person'], ConversationSummarizer.sbert_punc_case_model.punctuate(row['message'])]), axis=1).values)
        return input_text
    
    def summarize_conversation(self, conversation_df, conversation_id):
        input_text = self.add_puctuation_in_text(conversation_df, conversation_id)
        input_ids = ConversationSummarizer.tokenizer.encode(input_text, return_tensors='pt').to(ConversationSummarizer.device)
        outputs = ConversationSummarizer.model.generate(
            input_ids,
            eos_token_id=ConversationSummarizer.tokenizer.eos_token_id,
            num_beams=1,
            min_length=40,
            max_length=400,
            do_sample=True,
            no_repeat_ngram_size=4,
            top_p=0.8
        )
        result = ConversationSummarizer.tokenizer.decode(outputs[0][1:]).replace('</s>', '')
        return result
