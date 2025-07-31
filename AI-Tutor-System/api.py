# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database import get_db_context
from models import Interaction

app = FastAPI()

# Allow frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/conversation/{user_id}")
def get_conversation(user_id: str):
    with get_db_context() as db:
        # Get all interactions for the user, ordered by turn_number
        interactions = (
            db.query(Interaction)
            .filter(Interaction.user_id == user_id)
            .order_by(Interaction.turn_number.asc())
            .all()
        )
        if not interactions:
            raise HTTPException(status_code=404, detail="No conversation found for this user_id.")

        # Format for frontend: role, content, timestamp
        messages = []
        for interaction in interactions:
            # Student messages: use prompt, Tutor messages: use response
            messages.append({
                "role": "student",
                "content": interaction.prompt,
                "timestamp": interaction.timestamp.isoformat()
            })
            messages.append({
                "role": "tutor",
                "content": interaction.response,
                "timestamp": interaction.timestamp.isoformat()
            })
        return messages