from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import openai
import google.generativeai as genai
import uvicorn
import io
from PIL import Image
import numpy as np

# Configure API keys (Replace these with actual keys)
openai.api_key = "your-openai-api-key"
genai.configure(api_key="your-gemini-api-key")

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_ai(query: Query):
    """Fetch responses from OpenAI and Gemini."""
    response_openai = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query.question}]
    )
    
    response_gemini = genai.generate_content(query.question)
    
    return {
        "openai_response": response_openai["choices"][0]["message"]["content"],
        "gemini_response": response_gemini.text
    }

@app.post("/upload-image")
def upload_image(file: UploadFile = File(...)):
    """Process an uploaded image and return a response."""
    image = Image.open(io.BytesIO(file.file.read()))
    np_image = np.array(image)

    processed_text = "Image processing feature will be implemented."
    
    return {"message": processed_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
