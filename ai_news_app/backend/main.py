from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Import modules (to be implemented)
# from . import scraper, summarizer, emailer

app = FastAPI()

# CORS setup
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:5500", # For VS Code Live Server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    email: str
    news_items: List[dict]

@app.get("/")
def read_root():
    return {"message": "AI News Aggregator API is running"}

@app.get("/fetch-news")
def fetch_news():
    # TODO: Implement scraping logic
    return {"news": [{"title": "Test News", "summary": "This is a test summary", "link": "http://example.com"}]}

@app.post("/send-email")
def send_email(request: EmailRequest):
    # TODO: Implement email logic
    return {"message": f"Email sent to {request.email}"}
