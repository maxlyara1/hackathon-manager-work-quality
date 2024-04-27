import os

def extract_conversation(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        phrases = []
        for line in lines:
            line = line.strip()
            phrases.append(line)
    return phrases

def extract_data(conversation, conversation_id):
    data = []
    for phrase in conversation:
        person = phrase.split(":")[0].strip().capitalize()
        message = phrase.split(":")[1].strip()
        data.append((conversation_id, person, message))
    return data

table_data = []
for conversation_id in range(1, 14):
    file_path = f"/kaggle/input/manager-analysis/{conversation_id}.txt"
    conversation = extract_conversation(file_path)
    table_data.extend(extract_data(conversation, conversation_id))

df = pd.DataFrame(table_data, columns = ['conversation_id', 'person', 'message'])
df['message'] = df.groupby((df['person'] != df['person'].shift()).cumsum())['message'].transform(lambda x: ' '.join(x))
df = df.drop_duplicates(subset=['message', 'conversation_id','person'])
df