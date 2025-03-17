import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nltk.download('punkt')

# Voice Recognition Class
class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def start_listening(self):
        with sr.Microphone() as source:
            print("Listening...")
            try:
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                user_input.insert(tk.END, text)  # Insert recognized text into input box
            except sr.UnknownValueError:
                chat_output.insert(tk.END, "Bot: Sorry, I couldn't understand. Try again.\n\n", "bot")
            except sr.RequestError:
                chat_output.insert(tk.END, "Bot: Could not connect to recognition service.\n\n", "bot")

# Initialize Voice Recognition
voice_recognizer = VoiceRecognizer()

# FAQ Data
faq_data = {
    "What are your business hours?": "We are open from 9 AM to 5 PM, Monday to Friday.",
    "What is your return policy?": "You can return items within 30 days of purchase with a receipt.",
    "Do you offer international shipping?": "Yes, we ship to selected international locations.",
    "How can I contact customer support?": "You can reach us at support@example.com or call 123-456-7890.",
    "Where is your store located?": "Our store is located at 123 Main Street, Cityville.",
    "Hi": "Hello! How can I assist you today?",
    "Hello": "Hi there! How can I help you?",
    "How are you?": "I'm just a chatbot, but I'm here to help!",
    "What payment methods do you accept?": "We accept credit cards, PayPal, and Apple Pay.",
    "Do you offer discounts?": "Yes, we offer seasonal discounts. Stay tuned for promotions!",
    "Can I track my order?": "Yes, you can track your order using the tracking number provided after purchase.",
    "Do you have a mobile app?": "Yes, we have a mobile app available on iOS and Android.",
    "What is your email address?": "You can contact us at support@example.com.",
    "What is your phone number?": "You can reach us at 123-456-7890.",
    "Where do you ship?": "We ship worldwide to most countries.",
    "Can I change my shipping address after placing an order?": "Yes, you can modify your address within 24 hours of placing an order.",
    "What happens if my package is lost?": "If your package is lost, please contact customer support for assistance.",
    "Do you have a loyalty program?": "Yes, we have a rewards program where you earn points for every purchase.",
    "What is your refund policy?": "We provide full refunds within 30 days of purchase if the item is in its original condition."
}


questions = list(faq_data.keys())
answers = list(faq_data.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_best_match(user_input):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, X).flatten()
    best_match_index = np.argmax(similarities)
    best_score = similarities[best_match_index]
    
    if best_score > 0.2:
        return questions[best_match_index]
    else:
        return None

def chatbot_response(event=None):
    user_question = user_input.get("1.0", tk.END).strip()
    if not user_question:
        return
    
    best_match = get_best_match(user_question)
    response = faq_data.get(best_match, "Sorry, I couldn't find an exact match. Please try rephrasing.")
    
    chat_output.insert(tk.END, f"You: {user_question}\n", "user")
    chat_output.insert(tk.END, f"Bot: {response}\n\n", "bot")
    user_input.delete("1.0", tk.END)

def clear_chat():
    chat_output.delete("1.0", tk.END)

# GUI Setup
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("600x400")
root.configure(bg="#1E1E1E")

chat_output = scrolledtext.ScrolledText(root, height=15, width=70, bg="#2E2E2E", fg="white", font=("Arial", 12))
chat_output.pack(pady=10)
chat_output.tag_config("user", foreground="#4CAF50", font=("Arial", 12, "bold"))
chat_output.tag_config("bot", foreground="#00BFFF", font=("Arial", 12))

user_input = tk.Text(root, height=2, width=60, bg="#2E2E2E", fg="white", font=("Arial", 12))
user_input.pack(pady=5)
user_input.bind("<Return>", chatbot_response)

button_frame = tk.Frame(root, bg="#1E1E1E")
button_frame.pack()

tk.Button(button_frame, text="Send", command=chatbot_response, bg="#4CAF50", fg="white", font=("Arial", 12)).pack(side="left", padx=5)
tk.Button(button_frame, text="Clear", command=clear_chat, bg="#D32F2F", fg="white", font=("Arial", 12)).pack(side="left", padx=5)

speak_button = tk.Button(button_frame, text="🎙️ Speak", command=voice_recognizer.start_listening, bg="#FFA500", fg="white", font=("Arial", 12))
speak_button.pack(side="left", padx=5)

root.mainloop()
