import yagmail
import yaml
import streamlit as st
from yaml.loader import SafeLoader

class EmailService:
    def __init__(self, config_path='config.yaml'):
        # 1. Try Streamlit Secrets first (Production)
        if "smtp" in st.secrets:
            self.smtp_config = st.secrets["smtp"]
            self.sender = self.smtp_config.get('sender_email')
            self.password = self.smtp_config.get('app_password')
            self.app_url = self.smtp_config.get('app_url', '') # Capture app_url if available
        else:
            # 2. Fallback to local config.yaml (Local Dev)
            with open(config_path) as file:
                self.config = yaml.load(file, Loader=SafeLoader)
            
            self.smtp_config = self.config.get('smtp', {})
            self.sender = self.smtp_config.get('sender_email')
            self.password = self.smtp_config.get('app_password')
            self.app_url = self.smtp_config.get('app_url', 'http://localhost:8501') # Default logic
        
        if not self.sender or not self.password:
            # print("⚠️ SMTP credentials missing") # Quiet fail or log
            self.yag = None
        else:
            self.yag = yagmail.SMTP(user=self.sender, password=self.password)

    def send_email(self, to, subject, contents):
        """
        Send a single email.
        """
        if not self.yag:
            return False, "SMTP Configuration Error"
        
        try:
            self.yag.send(to=to, subject=subject, contents=contents)
            return True, "Email Sent Successfully"
        except Exception as e:
            return False, str(e)

    def send_bulk_email(self, recipient_list, subject, contents):
        """
        Send emails to a list of recipients.
        Uses BCC to avoid exposing emails if sending as a single batch, 
        or sends individually loop if personalizing (here we just loop for safety/simplicity).
        """
        if not self.yag:
            return False, "SMTP Configuration Error"
            
        success_count = 0
        failed_count = 0
        
        # Looping to avoid BCC limits or issues, and to allow for potential future personalization
        for recipient in recipient_list:
            try:
                self.yag.send(to=recipient, subject=subject, contents=contents)
                success_count += 1
            except Exception as e:
                print(f"Failed to send to {recipient}: {e}")
                failed_count += 1
                
        return True, f"Sent: {success_count}, Failed: {failed_count}"
