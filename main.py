import os
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from schemas import ClassifyRequest, ClassifyResponse
from classifier import MessageClassifier

# Initialize the FastAPI app
app = FastAPI(
    title="Telegram Message Value Classifier API",
    description="A lightweight, fast, and configurable rule-based API to detect high-value Telegram messages.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define path for rules configuration
RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules.yaml")

# Initialize classifier on startup
try:
    classifier = MessageClassifier(RULES_PATH)
except Exception as e:
    # Fallback/logging if file not found at startup
    classifier = None
    print(f"Error loading rules at startup: {e}")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Checks the status of the service and whether rules have been successfully loaded.
    """
    return {
        "status": "ok",
        "rules_loaded": classifier is not None and bool(classifier.rules)
    }


@app.post("/classify", response_model=ClassifyResponse, status_code=status.HTTP_200_OK)
async def classify_message(payload: ClassifyRequest):
    """
    Classifies a Telegram message based on its text, group/chat title, and sender name.
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Classifier is not initialized. Please ensure rules.yaml exists and is valid."
        )
    
    start_time = time.perf_counter()
    try:
        response = classifier.classify(
            raw_text=payload.text,
            chat_title=payload.chat_title,
            sender_name=payload.sender_name
        )
        
        # Calculate and log latency to meet performance criteria (<10ms)
        latency_ms = (time.perf_counter() - start_time) * 1000
        print(f"Classification completed in {latency_ms:.2f} ms")
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during classification: {str(e)}"
        )


@app.post("/reload-rules", status_code=status.HTTP_200_OK)
def reload_rules():
    """
    Hot-reloads the classification rules from the rules.yaml file.
    """
    global classifier
    try:
        if classifier is None:
            classifier = MessageClassifier(RULES_PATH)
        else:
            classifier.load_rules()
        return {
            "status": "ok",
            "message": "Rules reloaded successfully.",
            "rules_count": {
                "positive_categories": list(classifier.rules.get("positive_categories", {}).keys()),
                "spam_keywords_count": len(classifier.rules.get("fast_reject", {}).get("spam_keywords", []))
            }
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to reload rules: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
