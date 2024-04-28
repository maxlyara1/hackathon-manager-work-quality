from call_data_preparation import ConversationExtractor
from call_summary import ConversationSummarizer
from manager_help_detection import ManagerHelped
from manager_sentiment_analysis import SentimentAnalyzer
from manager_support_mark import SetMarkForManager
import pandas as pd
import streamlit as st
import os
import numpy as np
import plotly.graph_objects as go


def get_text(conversation_df, conversation_id, persons, speaker=True):
    if speaker:
        input_text = ' '.join(conversation_df[(conversation_df['conversation_id'] == conversation_id) & (conversation_df['person'].isin(persons))].apply(lambda row: ': '.join([row['person'], ConversationSummarizer.sbert_punc_case_model.punctuate(row['message'])]), axis=1).values)
        return input_text
    else:
        input_text = ' '.join(conversation_df[(conversation_df['conversation_id'] == conversation_id) & (conversation_df['person'].isin(persons))].apply(lambda row: ConversationSummarizer.sbert_punc_case_model.punctuate(row['message']), axis=1).values)
        return input_text

def main():

    polite_count = 0
    not_polite_count = 0

    manager_helped_count = 0
    manager_not_helped_count = 0

    os.makedirs("temp", exist_ok=True)
    st.title("Conversations Data")

    st.sidebar.header("Upload Files")
    uploaded_files = st.sidebar.file_uploader("Upload TXT files", type="txt", accept_multiple_files=True)

    if uploaded_files:
        table_data = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            conversation_extractor = ConversationExtractor(file_path)
            conversation = conversation_extractor.extract_conversation()
            table_data.extend(conversation_extractor.extract_data(conversation, file_path))

        df = pd.DataFrame(table_data, columns=['conversation_id', 'person', 'message'])

        df['message'] = df.groupby(['conversation_id', (df['person'] != df['person'].shift()).cumsum()])['message'].transform(lambda x: ' '.join(x))

        df = df.drop_duplicates(subset=['message', 'conversation_id', 'person']).reset_index(drop=True)
        st.write(df)
        st.markdown("""---""")

        summarizer = ConversationSummarizer()
        for conversation_id in np.unique(df['conversation_id'].values):

            st.write(f'conversation_id - {conversation_id}: ')
            manager_help_model = ManagerHelped()
            text_old_client = ' '.join(df[(df['conversation_id'] == conversation_id) & (df['person'].isin(['Клиент']))]['message'].values)
            text_old_manager = ' '.join(df[(df['conversation_id'] == conversation_id) & (df['person'].isin(['Оператор', 'Сотрудник', 'Менеджер']))]['message'].values)
            is_manager_helped = manager_help_model.analyze_manager_text(text_old_client)
            st.write(f'Manager helped: {is_manager_helped}')
            if is_manager_helped:
                manager_helped_count += 1
            else:
                manager_not_helped_count += 1

            sentiment_analyzer = SentimentAnalyzer()
            is_manager_was_polite = sentiment_analyzer.analyze_text_sentiment(text_old_manager)
            st.write(f'Manager was polite: {is_manager_was_polite}')
            if is_manager_was_polite:
                polite_count += 1
            else:
                not_polite_count += 1

            set_mark_model = SetMarkForManager()
            mark = set_mark_model.result(get_text(df, conversation_id, persons=['Оператор', 'Сотрудник', 'Менеджер'], speaker=False), get_text(df, conversation_id, persons=['Клиент'], speaker=False))
            st.write(f"Mark: {mark}")


            if st.button(f"Summarize {conversation_id}"):
                st.write(f'{conversation_id} summary:')
                conversation_summary = summarizer.summarize_conversation(df, conversation_id)
                st.write(conversation_summary)

            if st.button(f'Look at text of {conversation_id}'):
                text_with_punctuation_all = '\n'.join(df[(df['conversation_id'] == conversation_id) & (df['person'].isin(['Оператор', 'Сотрудник', 'Менеджер','Клиент']))].apply(lambda row: ': '.join([row['person'], ConversationSummarizer.sbert_punc_case_model.punctuate(row['message'])]), axis=1).values)+'\n'
                st.text_area(label=f'{conversation_id} text:', value=text_with_punctuation_all.replace('\n', '\n\n'), height=600, on_change=False)

            st.markdown("""---""")

        # Visualization using Plotly
        st.write("Pie Chart of Politeness")
        polite_labels = ['Polite', 'Not Polite']
        polite_values = [polite_count, not_polite_count]
        polite_fig = go.Figure(data=[go.Pie(labels=polite_labels, values=polite_values)])
        st.plotly_chart(polite_fig)

        # Visualization using Plotly
        st.write("Pie Chart of Assistance of Manager")
        manager_help_labels = ['Manager helped', 'Manager not helped']
        manager_help_values = [manager_helped_count, manager_not_helped_count]
        manager_help_fig = go.Figure(data=[go.Pie(labels=manager_help_labels, values=manager_help_values)])
        st.plotly_chart(manager_help_fig)

        st.markdown("""---""")

    else:
        st.write('There are no uploaded files yet')

if __name__ == "__main__":
    main()
