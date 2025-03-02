# src/utils.py

import pandas as pd
from sbert_punc_case_ru import SbertPuncCase

_sbert_punc_case_model = SbertPuncCase()


def get_text(
    conversation_df: pd.DataFrame,
    conversation_id: str,
    persons: list[str],
    speaker: bool = True,
) -> str:
    """
    Извлекает текст из DataFrame.
    """
    if speaker:
        return " ".join(
            f"{row['person']}: {_sbert_punc_case_model.punctuate(row['message'])}"
            for _, row in conversation_df[
                (conversation_df["conversation_id"] == conversation_id)
                & (conversation_df["person"].isin(persons))
            ].iterrows()
        )
    else:
        return " ".join(
            _sbert_punc_case_model.punctuate(row["message"])
            for _, row in conversation_df[
                (conversation_df["conversation_id"] == conversation_id)
                & (conversation_df["person"].isin(persons))
            ].iterrows()
        )
