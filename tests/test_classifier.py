import os
import pytest
from fastapi.testclient import TestClient

# We need to add the parent directory to sys.path so we can import modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, RULES_PATH
from classifier import MessageClassifier
from normalizer import normalize_text

client = TestClient(app)

@pytest.fixture
def classifier():
    return MessageClassifier(RULES_PATH)


def test_normalizer():
    # Test Arabic replacement, lowercase, and spaces/diacritics
    text = "سلام، شركت ما نيازمندِ  برنامه نويسِ پايتون است."
    normalized = normalize_text(text)
    # ك -> ک, ي -> ی, duplicate space removed, diacritic removed
    assert "شرکت" in normalized
    assert "نیازمند" in normalized
    assert "نویس" in normalized
    assert "پایتون" in normalized
    assert "  " not in normalized


def test_hiring_message(classifier):
    # Valid hiring message
    text = (
        "سلام وقت بخیر. شرکت ما استخدام می‌کند. دعوت به همکاری برای جذب نیرو برنامه نویس "
        "python و django داریم. دورکاری و تمام وقت. لطفا رزومه خود را ارسال کنید."
    )
    result = classifier.classify(text)
    assert result.valuable is True
    assert result.category == "hiring"
    assert result.score >= 5
    assert "hiring" in result.reasons[0]


def test_project_request_message(classifier):
    # Valid project request message
    text = (
        "سلام. یک پروژه دارم برای ساخت ربات تلگرام. سفارش پروژه طراحی سایت با ری‌اکت "
        "و جنگو داریم. نیاز به فریلنسر دارم. هزینه و زمان تحویل هم توافقی است."
    )
    result = classifier.classify(text)
    assert result.valuable is True
    assert result.category == "project_request"
    assert result.score >= 5


def test_collaboration_message(classifier):
    # Valid collaboration/startup partnership message
    text = (
        "سلام به همگی. ما یک استارتاپ جدید در حوزه هوش مصنوعی هستیم و به دنبال هم بنیانگذار "
        "و شریک فنی برای شراکت و کار تیمی هستیم. اگر مایل به همکاری هستید پیام بدید."
    )
    result = classifier.classify(text)
    assert result.valuable is True
    assert result.category == "collaboration"
    assert result.score >= 5


def test_crypto_spam_message(classifier):
    # Spams should be rejected quickly
    text = (
        "کازینو آنلاین و شرط بندی فوتبال با بونوس ویژه! سیگنال‌های ترید روزانه فارکس "
        "و ارز دیجیتال در کانال ما. همین حالا کلیک کنید."
    )
    result = classifier.classify(text)
    assert result.valuable is False
    assert result.category == "irrelevant"


def test_link_only_message(classifier):
    # Messages containing only links must be rejected
    text = "https://t.me/some_channel_link_example_test"
    result = classifier.classify(text)
    assert result.valuable is False
    assert result.category == "irrelevant"


def test_short_and_vague_message(classifier):
    # Short messages under 20 chars should be rejected
    text = "سلام کسی هست؟"
    result = classifier.classify(text)
    assert result.valuable is False
    assert result.category == "irrelevant"


def test_advertising_message(classifier):
    # Marketing and general product sales should be rejected
    text = (
        "فروش فوق العاده و حراج انواع کفش‌های چرم تبریز! تخفیف ویژه ۵۰ درصدی برای ثبت نام "
        "و خرید محصول از وبسایت ما. ارسال فوری به کل کشور."
    )
    result = classifier.classify(text)
    assert result.valuable is False
    # Should get flagged by negative indicators or fast rejection
    assert result.score <= 0 or result.valuable is False


def test_arabic_characters_message(classifier):
    # Test Arabic characters normalization in a real hiring context
    text = "سلام، شركت ما نيازمند برنامه نويس پايتون است. ارسال رزومه براى استخدام."
    result = classifier.classify(text)
    assert result.valuable is True
    assert result.category == "hiring"


# --- API Endpoint Integration Tests ---

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["rules_loaded"] is True


def test_api_classify_valid():
    payload = {
        "text": "فرصت شغلی عالی! نیازمند یک برنامه نویس فرانت اند مسلط به ری‌اکت به صورت دورکاری و تمام وقت.",
        "chat_title": "Telegram Job Board",
        "sender_name": "HR Manager"
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valuable"] is True
    assert data["category"] == "hiring"
    assert data["confidence"] >= 0.5


def test_api_classify_invalid():
    payload = {
        "text": "سلام به همه اعضا 🌸 صبح قشنگتون بخیر باشه.",
        "chat_title": "General Chat"
    }
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valuable"] is False
    assert data["category"] == "irrelevant"


def test_api_reload_rules():
    response = client.post("/reload-rules")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "positive_categories" in response.json()["rules_count"]
