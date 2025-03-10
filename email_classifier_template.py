import os
import json
import pandas as pd
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging
from sklearn.metrics import accuracy_score


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
    },

    # Edge case: email with insufficient content
       {
        "id": "006",
        "from": "business@example.com",
        "subject": "Hola",
        "body": ""
    }, 
    # Edge case for mixed category
     {
        "id": "007",
        "from": "Partheclient@example.com",
        "subject": "Congratulations and Partnership opportunity",
        "body": "I want to express my gratitude for the excellent support I received from Sarah on your team. She went above and beyond to resolve my issue—keep up the great work! Additionally, our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-06-15T15:021:00Z"
    },
    # Edge case for random email
{
    "id": "008",
    "from": "Crazy@example.com",
    "subject": "Hi mate",
    "body": "Just to mention I'm the king of the world",

    }

]
# Accuracy of the classification model
true_labels = {
    "001": "complaint",
    "002": "inquiry",
    "003": "feedback",
    "004": "support_request",
    "005": "other",
    "006": "other",
    "007": "other",
    "008": "other"

}

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
        # Edge case for emails with insufficient content
        if not email.get("body") or len(email["body"].strip()) < 5:
            logger.warning(f"Email {email['id']} has insufficient content.")
            return "other" 
        
        
        prompt = f"""
        You are an AI that classifies emails into one or more of the following categories:
        - complaint (e.g., "My product arrived broken, I want a refund.")
        - inquiry (e.g., "Does this software work on Mac OS?")
        - feedback (e.g., "Great customer service, I'm very satisfied!")
        - support_request (e.g., "I'm getting error 5123 while installing the app.")
        - other (e.g., "We'd like to explore a partnership opportunity.")

        Rules:
        1. If an email belongs to multiple categories, return other.
        2. If uncertain, return other.

        Here is an email:
        Subject: {email["subject"]}
        Body: {email["body"]}

        Please return ONLY the category that best describes this email.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "system", "content": "You are an AI assistant that classifies emails in complaint, inquiry, feedback, support_request or other."},
                {"role": "user", "content": prompt}],
                max_tokens=10,
                #temperature=0
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

class EmailAutomationSystem:
    def __init__(self, processor: EmailProcessor):
        """Initialize the automation system with an EmailProcessor."""
        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

    def process_emails(self, emails: List[Dict]) -> pd.DataFrame:
        """
        Processes a list of emails: classifies each email and generates a response.
        Returns a DataFrame with the results.
        """
        results = []

        for email in emails:
            logger.info(f"\nProcessing email {email['id']}...")

            # Classify the email
            classification = self.processor.classify_email(email)
            logger.info(f"Classification: {classification}")

            # Generate a response
            response = self.processor.generate_response(email, classification) 
            logger.info(f"Generated response: {response}")

            results.append({
                "email_id": email["id"],
                "success": classification != "unknown",
                "classification": classification,
                "response_sent": response is not None
            })

        return pd.DataFrame(results)
    
    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        TODO: Implement complaint handling logic
        """
        pass

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        TODO: Implement inquiry handling logic
        """
        pass

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        TODO: Implement feedback handling logic
        """
        pass

    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        pass

    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        TODO: Implement handling logic for other categories
        """
        pass

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}")
    # In real implementation: integrate with email service


def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}")
    # In real implementation: integrate with feedback system

# Execute email processing
if __name__ == "__main__":
    processor = EmailProcessor()  
    automation_system = EmailAutomationSystem(processor)
    results_df = automation_system.process_emails(sample_emails)

    print("\nProcessing completed:")
    print(results_df)

# Predicted labels and actual labels
predicted_labels = results_df["classification"].tolist()
actual_labels = [true_labels[email_id] for email_id in results_df["email_id"].tolist()]

# calculate accuracy
accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Classification Accuracy: {accuracy:.2f}")

# Display misclassified emails
for idx, (pred, actual) in enumerate(zip(predicted_labels, actual_labels)):
    if pred != actual:
        print(f"Misclassified email {results_df.loc[idx, 'email_id']}: predicted '{pred}', actual '{actual}'")
