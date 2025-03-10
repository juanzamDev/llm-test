import os
import json
import pandas as pd
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample email dataset
sample_emails = [
    {
        "id": "001",
        "from": "angry.customer@example.com",
        "subject": "Broken product received",
        "body": "I received my order #12345 yesterday but it arrived completely damaged. This is unacceptable and I demand a refund immediately. This is the worst customer service I've experienced.",
        "timestamp": "2024-03-15T10:30:00Z"
    },
    {
        "id": "002",
        "from": "curious.shopper@example.com",
        "subject": "Question about product specifications",
        "body": "Hi, I'm interested in buying your premium package but I couldn't find information about whether it's compatible with Mac OS. Could you please clarify this? Thanks!",
        "timestamp": "2024-03-15T11:45:00Z"
    },
    {
        "id": "003",
        "from": "happy.user@example.com",
        "subject": "Amazing customer support",
        "body": "I just wanted to say thank you for the excellent support I received from Sarah on your team. She went above and beyond to help resolve my issue. Keep up the great work!",
        "timestamp": "2024-03-15T13:15:00Z"
    },
    {
        "id": "004",
        "from": "tech.user@example.com",
        "subject": "Need help with installation",
        "body": "I've been trying to install the software for the past hour but keep getting error code 5123. I've already tried restarting my computer and clearing the cache. Please help!",
        "timestamp": "2024-03-15T14:20:00Z"
    },
    {
        "id": "005",
        "from": "business.client@example.com",
        "subject": "Partnership opportunity",
        "body": "Our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-03-15T15:00:00Z"
    }
]

class EmailProcessor:
    def __init__(self):
        """Initializes the email processor with OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define valid email categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Uses OpenAI to classify an email into one of the predefined categories.
        Returns the category or None if classification fails.
        """
        prompt = f"""
        Categorize the following email into one of these categories: 
        complaint, inquiry, feedback, support_request, other.
        Email:
        Subject: {email["subject"]}
        Body: {email["body"]}
        Respond with only the category.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=10
            )

            category = response.choices[0].message.content.strip().lower()

            # Ensure the category is valid
            if category in self.valid_categories:
                return category
            else:
                return "other"

        except Exception as e:
            logger.error(f"Error classifying email: {e}")
            return "unknown"

    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generates an automatic response based on the classified email category.
        """
        response_templates = {
            "complaint": "We sincerely apologize for the inconvenience. Our team will review your case and respond shortly.",
            "inquiry": "Thank you for your inquiry. We will get back to you with the requested information as soon as possible.",
            "feedback": "We truly appreciate your feedback. It helps us improve our services.",
            "support_request": "We understand your issue, and our support team will contact you soon.",
            "other": "Thank you for your message. We will review it and respond shortly."
        }

        return response_templates.get(classification, "Thank you for your message.")

    def process_emails(self, emails: List[Dict]) -> pd.DataFrame:
        """
        Processes a list of emails: classifies each email and generates a response.
        Returns a DataFrame with the results.
        """
        results = []

        for email in emails:
            logger.info(f"\nProcessing email {email['id']}...")

            # Classify the email
            classification = self.classify_email(email)
            logger.info(f"Classification: {classification}")

            # Generate a response
            response = self.generate_response(email, classification)
            logger.info(f"Generated response: {response}")

            results.append({
                "email_id": email["id"],
                "success": classification != "unknown",
                "classification": classification,
                "response_sent": response is not None
            })

        return pd.DataFrame(results)

# Execute email processing
if __name__ == "__main__":
    processor = EmailProcessor()
    results_df = processor.process_emails(sample_emails)

    print("\nProcessing completed:")
    print(results_df)
