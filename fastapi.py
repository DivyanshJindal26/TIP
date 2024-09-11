from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request model for processing the word
class WordModel(BaseModel):
    word: str

# Endpoint to process the current word
@app.post("/process")
def process_word(word_model: WordModel):
    word = word_model.word
    # Add logic to process the word (e.g., translation, etc.)
    processed_output = word.upper()  # Example transformation
    return {"output": processed_output}
