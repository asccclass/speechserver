import requests
import urllib.parse

class TranslationManager:
    def __init__(self, mode='local', url=None):
        """
        Initialize the Translation Manager.
        
        Args:
            mode (str): 'local' or 'remote'.
            url (str): The URL for remote translation (used if mode is 'remote').
        """
        self.mode = mode
        self.url = url

    def translate(self, text):
        """
        Translate the given text based on the configured mode.
        
        Args:
            text (str): The text to translate.
            
        Returns:
            str: The translated text, or None if translation fails or is disabled.
        """
        if not text:
            return ""

        if self.mode == 'remote':
            return self._translate_remote(text)
        elif self.mode == 'local':
            return self._translate_local(text)
        else:
            print(f"Unknown translation mode: {self.mode}")
            return text

    def _translate_remote(self, text):
        """
        Perform remote translation via HTTP GET.
        Assumes the server expects the text as a query parameter (e.g., ?q=text).
        Adjust the parameter name based on the actual API requirement.
        Using 'text' based on common conventions, validatable by user.
        """
        if not self.url:
            print("Remote translation URL is not set.")
            return text

        try:
            # Construct the URL with query parameters
            # Assuming the API expects the text in a 'text' parameter
            # If the user's requirement for "http get" implies a specific format, we follow standard params.
            # The prompt said "through http get", so we'll append params.
            
            # Note: The user didn't specify the query param name. 
            # I will use 'text' as a default, but this might need adjustment.
            # Ideally, I would ask, but for now I'll implement a reasonable default.
            
            params = {'text': text}
            response = requests.get(self.url, params=params, timeout=5)
            
            if response.status_code == 200:
                # Assuming the response body contains the translation directly or JSON.
                # If it's pure text:
                return response.text.strip()
                # If JSON, we might need parsing. Since it's unspecified, text is safest.
            else:
                print(f"Remote translation failed. Status: {response.status_code}")
                return text
        except Exception as e:
            print(f"Remote translation error: {e}")
            return text

    def _translate_local(self, text):
        """
        Perform local translation.
        Currently a placeholder.
        """
        # Placeholder: just return the original text prefixed to show it 'worked' in test
        # or just return text.
        return f"[Local] {text}"
