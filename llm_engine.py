import requests
import json
import threading


class OllamaLLM:
    """
    Handles communication with a self-hosted Ollama LLM server
    (e.g., qwen2.5-coder:32b running via ngrok tunnel).
    """

    def __init__(self, base_url="", model_name="qwen2.5-coder:32b"):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.is_connected = False
        self.headers = {"ngrok-skip-browser-warning": "true"}

    def set_config(self, base_url, model_name):
        """Update server URL and model name at runtime."""
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name

    def check_connection(self):
        """
        Pings the Ollama server root endpoint to verify it is online.
        Returns (is_connected: bool, message: str).
        """
        if not self.base_url:
            self.is_connected = False
            return False, "Server URL is empty. Please enter your Ollama/ngrok URL in Settings."

        try:
            r = requests.get(
                f"{self.base_url}/",
                headers=self.headers,
                timeout=8
            )
            if r.status_code == 200 and "running" in r.text.lower():
                self.is_connected = True
                return True, "Ollama is running and connected!"
            else:
                self.is_connected = False
                return False, f"Server responded with status {r.status_code}, but Ollama may not be running."
        except requests.exceptions.Timeout:
            self.is_connected = False
            return False, "Connection timed out. Server may be offline or URL is incorrect."
        except requests.exceptions.ConnectionError:
            self.is_connected = False
            return False, "Connection refused. Make sure the Ollama server and ngrok tunnel are running."
        except Exception as e:
            self.is_connected = False
            return False, f"Connection error: {str(e)}"

    def _build_system_prompt(self, faq_list):
        """
        Builds a system instruction prompt embedding the FAQ database
        as the chatbot's knowledge base context with multilingual capability.
        """
        prompt = (
            "You are a friendly, helpful, and highly accessible AI FAQ Assistant. "
            "Your goal is to make answers easy to understand for everyone, including beginners and non-technical people.\n\n"
            "=== MULTILINGUAL & COMMUNICATION RULES (CRITICAL) ===\n"
            "1. AUTOMATIC LANGUAGE MATCHING:\n"
            "   - If the user talks in or asks for Hinglish / Roman Urdu (e.g. 'kya hal hai', 'hinglish me baat kro', 'order kaisy hoga', 'urdu me bolo', etc.), "
            "     you MUST respond in natural, friendly, conversational Hinglish (Roman Urdu).\n"
            "   - If the user talks in Urdu script (e.g. 'آپ کیسے ہیں؟'), respond in clean, natural Urdu.\n"
            "   - If the user talks in English, respond in clear, simple English.\n"
            "   - If the user asks to switch language (e.g. 'speak in Hinglish', 'English me bolo', 'Urdu please'), immediately switch to that language and acknowledge pleasantly.\n"
            "2. SIMPLICITY:\n"
            "   - Explain concepts in simple, easy-to-understand words. Avoid overly complex technical jargon unless asked.\n"
            "3. KNOWLEDGE USAGE:\n"
            "   - Use the knowledge base below to answer questions accurately. If answering in Hinglish or Urdu, translate and adapt the knowledge naturally so it sounds human and conversational.\n"
            "   - If a question is about casual conversation or not in the knowledge base, respond politely, helpfully, and naturally.\n\n"
            "=== KNOWLEDGE BASE (FAQ Database) ===\n"
        )

        if faq_list:
            for i, faq in enumerate(faq_list, 1):
                prompt += f"\nQ{i}: {faq['question']}\nA{i}: {faq['answer']}\n"
        else:
            prompt += "\n(No FAQs currently loaded)\n"

        prompt += "\n=== END OF KNOWLEDGE BASE ===\n\n"
        prompt += "Now answer the user's question following the language and tone rules above:\n\n"

        return prompt

    def generate_response(self, user_query, faq_list, on_token_callback=None):
        """
        Sends the user query (with FAQ context) to the Ollama API.
        Streams the response token-by-token via on_token_callback.

        Args:
            user_query (str): The user's question.
            faq_list (list): Current FAQ database list of dicts.
            on_token_callback (callable): Called with each token string as it arrives.

        Returns:
            (full_response: str, success: bool, error_msg: str)
        """
        if not self.base_url:
            return "", False, "Server URL is not configured. Go to Settings to set it up."

        system_prompt = self._build_system_prompt(faq_list)
        full_prompt = system_prompt + f"User: {user_query}\nAssistant:"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": True
                },
                stream=True,
                timeout=120
            )

            if response.status_code != 200:
                return "", False, f"Server returned HTTP {response.status_code}."

            full_text = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        piece = chunk.get("response", "")
                        if piece:
                            full_text += piece
                            if on_token_callback:
                                on_token_callback(piece)
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

            return full_text.strip(), True, ""

        except requests.exceptions.Timeout:
            return "", False, "Request timed out. The model may be loading or the server is slow."
        except requests.exceptions.ConnectionError:
            return "", False, "Cannot connect to server. Make sure Ollama and ngrok are running."
        except Exception as e:
            return "", False, f"LLM error: {str(e)}"

    def generate_response_threaded(self, user_query, faq_list, on_token_callback=None, on_complete_callback=None):
        """
        Runs generate_response in a background thread so the GUI doesn't freeze.

        Args:
            on_token_callback: Called with each token (from background thread, use after() in GUI).
            on_complete_callback: Called with (full_response, success, error_msg) when done.
        """
        def _worker():
            full_text, success, error = self.generate_response(
                user_query, faq_list, on_token_callback
            )
            if on_complete_callback:
                on_complete_callback(full_text, success, error)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
