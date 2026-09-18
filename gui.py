import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import os
import random
import time
import threading
from faq_engine import FAQEngine, NLTK_AVAILABLE, SKLEARN_AVAILABLE
from llm_engine import OllamaLLM

# Configure default customtkinter behavior
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class FAQChatbotGUI(ctk.CTk):
    def __init__(self, engine: FAQEngine, default_db_path: str, llm_engine: OllamaLLM = None):
        super().__init__()
        
        self.engine = engine
        self.llm = llm_engine if llm_engine else OllamaLLM()
        self.default_db_path = default_db_path
        self.similarity_threshold = 0.30
        self.llm_mode_active = False
        self.is_llm_responding = False  # Lock to prevent double-sends while streaming
        self.active_language = "Auto Detect"
        
        # Configure window settings
        self.title("AI FAQ Chatbot - Professional Suite")
        self.geometry("1100x650")
        self.minimum_size = (900, 550)
        self.minsize(self.minimum_size[0], self.minimum_size[1])
        
        # Selected index in FAQ manager
        self.selected_faq_idx = None
        
        # Create layouts
        self._setup_layout()
        
        # Load initial FAQ database
        self._load_faq_database(default_db_path)
        
        # Select default frame
        self._select_frame("chat")
        
        # Welcome message in chat
        self._add_bot_message("Hello! I am your AI FAQ Assistant. How can I help you today? Ask me any question about general knowledge, technology, customer support, orders, or admissions!")
        self._refresh_quick_questions()

    def _setup_layout(self):
        """Creates the sidebar and container frames for tabs."""
        # Grid layout configuration (1 row, 2 columns)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        # 1. SIDEBAR FRAME
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)  # spacer row
        
        # Sidebar Logo
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="🤖 FAQ Chatbot", 
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 30))
        
        # Sidebar Buttons
        self.chat_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="💬 Chatbot", 
            fg_color="transparent", 
            text_color=("gray10", "gray90"), 
            hover_color=("gray70", "gray30"),
            anchor="w",
            font=ctk.CTkFont(size=14),
            command=lambda: self._select_frame("chat")
        )
        self.chat_btn.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        self.manager_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="📚 FAQ Database", 
            fg_color="transparent", 
            text_color=("gray10", "gray90"), 
            hover_color=("gray70", "gray30"),
            anchor="w",
            font=ctk.CTkFont(size=14),
            command=lambda: self._select_frame("manager")
        )
        self.manager_btn.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        
        self.settings_btn = ctk.CTkButton(
            self.sidebar_frame, 
            text="⚙️ Settings & Info", 
            fg_color="transparent", 
            text_color=("gray10", "gray90"), 
            hover_color=("gray70", "gray30"),
            anchor="w",
            font=ctk.CTkFont(size=14),
            command=lambda: self._select_frame("settings")
        )
        self.settings_btn.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        
        # Appearance Mode Selector in sidebar footer
        self.appearance_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance:", anchor="w", font=ctk.CTkFont(size=11))
        self.appearance_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_option = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=["System", "Light", "Dark"], 
            command=self._change_appearance_mode
        )
        self.appearance_option.grid(row=6, column=0, padx=20, pady=(5, 5), sticky="ew")
        
        # LLM Mode Toggle Switch
        self.llm_switch_label = ctk.CTkLabel(self.sidebar_frame, text="LLM Mode:", anchor="w", font=ctk.CTkFont(size=11))
        self.llm_switch_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.llm_switch = ctk.CTkSwitch(
            self.sidebar_frame,
            text="OFF",
            font=ctk.CTkFont(size=11),
            command=self._toggle_llm_mode,
            onvalue=1,
            offvalue=0
        )
        self.llm_switch.grid(row=8, column=0, padx=20, pady=(5, 5), sticky="w")
        
        self.llm_status_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text="FAQ Mode", 
            text_color="gray50",
            font=ctk.CTkFont(size=10)
        )
        self.llm_status_label.grid(row=9, column=0, padx=20, pady=(0, 15), sticky="w")
        
        # 2. MAIN FRAMES CONTAINER
        # Each tab is a separate Frame mapped to a dictionary
        self.frames = {}
        
        self._setup_chat_frame()
        self._setup_manager_frame()
        self._setup_settings_frame()

    def _select_frame(self, name):
        """Switches the active frame and highlights corresponding sidebar button."""
        # Reset colors of all buttons
        for btn in [self.chat_btn, self.manager_btn, self.settings_btn]:
            btn.configure(fg_color="transparent")
            
        # Highlight selected button and show frame
        if name == "chat":
            self.chat_btn.configure(fg_color=("gray75", "gray25"))
            self.frames["chat"].grid(row=0, column=1, sticky="nsew")
            self.frames["manager"].grid_forget()
            self.frames["settings"].grid_forget()
            self.entry_field.focus()
        elif name == "manager":
            self.manager_btn.configure(fg_color=("gray75", "gray25"))
            self.frames["manager"].grid(row=0, column=1, sticky="nsew")
            self.frames["chat"].grid_forget()
            self.frames["settings"].grid_forget()
            self._refresh_manager_list()
        elif name == "settings":
            self.settings_btn.configure(fg_color=("gray75", "gray25"))
            self.frames["settings"].grid(row=0, column=1, sticky="nsew")
            self.frames["chat"].grid_forget()
            self.frames["manager"].grid_forget()
            self._update_settings_stats()

    def _change_appearance_mode(self, new_mode):
        ctk.set_appearance_mode(new_mode)

    def _toggle_llm_mode(self):
        """Toggles between FAQ-only mode and LLM-powered mode."""
        if self.llm_switch.get() == 1:
            # Turning ON LLM mode
            if not self.llm.base_url:
                messagebox.showwarning(
                    "LLM Not Configured",
                    "Please go to Settings & Info tab and enter your Ollama server URL first."
                )
                self.llm_switch.deselect()
                return
            
            # Quick connection check
            connected, msg = self.llm.check_connection()
            if not connected:
                messagebox.showerror("Connection Failed", f"Cannot connect to Ollama server:\n{msg}")
                self.llm_switch.deselect()
                return
            
            self.llm_mode_active = True
            self.llm_switch.configure(text="ON")
            self.llm_status_label.configure(text="LLM Active", text_color="#2fa572")
            self._add_bot_message("LLM Mode activated! I am now powered by AI (Ollama). Ask me anything!")
        else:
            # Turning OFF LLM mode
            self.llm_mode_active = False
            self.llm_switch.configure(text="OFF")
            self.llm_status_label.configure(text="FAQ Mode", text_color="gray50")
            self._add_bot_message("Switched back to FAQ Mode. I will answer from the loaded FAQ database.")

    # =========================================================================
    # CHAT INTERFACE TAB
    # =========================================================================
    def _setup_chat_frame(self):
        chat_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frames["chat"] = chat_frame
        
        # 3 Rows: Header, Chat Messages Area, Input Area
        chat_frame.grid_rowconfigure(1, weight=1)
        chat_frame.grid_columnconfigure(0, weight=1)
        
        # Chat Header
        self.chat_header = ctk.CTkFrame(chat_frame, height=50, corner_radius=0, fg_color=("gray90", "gray15"))
        self.chat_header.grid(row=0, column=0, sticky="ew")
        self.chat_header.grid_columnconfigure(0, weight=1)
        
        self.chat_title = ctk.CTkLabel(
            self.chat_header, 
            text="FAQ Chatbot Assistant", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.chat_title.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        
        # Chat Header Controls (Language Selector + Clear Chat)
        ctrl_frame = ctk.CTkFrame(self.chat_header, fg_color="transparent")
        ctrl_frame.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        
        lang_lbl = ctk.CTkLabel(ctrl_frame, text="Language:", font=ctk.CTkFont(size=12))
        lang_lbl.pack(side="left", padx=(0, 6))
        
        self.lang_option = ctk.CTkOptionMenu(
            ctrl_frame,
            values=["Auto Detect", "English", "Hinglish (Roman Urdu)", "Urdu (اردو)"],
            width=150,
            height=26,
            command=self._on_language_changed
        )
        self.lang_option.set(self.active_language)
        self.lang_option.pack(side="left", padx=(0, 10))
        
        self.clear_btn = ctk.CTkButton(
            ctrl_frame, 
            text="Clear Chat", 
            width=80, 
            height=26, 
            fg_color="transparent", 
            border_width=1, 
            text_color=("gray10", "gray90"),
            command=self._clear_chat_log
        )
        self.clear_btn.pack(side="left")
        
        # Chat Messages Scrollable Frame
        self.chat_scroll = ctk.CTkScrollableFrame(chat_frame, corner_radius=0, fg_color="transparent")
        self.chat_scroll.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_scroll.grid_columnconfigure(0, weight=1)
        
        # Suggested questions panel (chips)
        self.suggestions_frame = ctk.CTkFrame(chat_frame, height=45, fg_color="transparent")
        self.suggestions_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 5))
        
        self.suggest_label = ctk.CTkLabel(self.suggestions_frame, text="Suggested:", font=ctk.CTkFont(size=11, weight="bold"))
        self.suggest_label.pack(side="left", padx=5)
        
        self.suggestion_buttons = []
        for i in range(3):
            btn = ctk.CTkButton(
                self.suggestions_frame, 
                text="", 
                height=25, 
                fg_color=("gray85", "gray25"),
                text_color=("gray10", "gray90"), 
                hover_color=("gray75", "gray35"),
                corner_radius=12,
                font=ctk.CTkFont(size=11),
                command=lambda b_idx=i: self._ask_suggestion(b_idx)
            )
            btn.pack(side="left", padx=5)
            self.suggestion_buttons.append(btn)
            
        # Bottom Input Area Frame
        self.input_frame = ctk.CTkFrame(chat_frame, fg_color="transparent")
        self.input_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=(0, 15))
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.entry_field = ctk.CTkEntry(
            self.input_frame, 
            placeholder_text="Ask a question about the product or topic...", 
            height=45,
            corner_radius=8
        )
        self.entry_field.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry_field.bind("<Return>", lambda e: self._send_user_message())
        self._bind_entry_context_menu(self.entry_field)
        
        self.send_btn = ctk.CTkButton(
            self.input_frame, 
            text="Send ➡️", 
            width=100, 
            height=45,
            corner_radius=8,
            command=self._send_user_message
        )
        self.send_btn.grid(row=0, column=1, sticky="ew")

    def _copy_to_clipboard(self, text, feedback_btn=None):
        """Copies text to system clipboard and provides button feedback."""
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
            if feedback_btn:
                orig_text = feedback_btn.cget("text")
                feedback_btn.configure(text="Copied! ✓")
                self.after(1500, lambda: feedback_btn.configure(text=orig_text))
        except Exception:
            pass

    def _bind_copy_menu(self, widget, text_or_getter):
        """Binds a right-click context menu with 'Copy' option to any widget."""
        def show_menu(event):
            text = text_or_getter() if callable(text_or_getter) else text_or_getter
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="📋 Copy Message", command=lambda: self._copy_to_clipboard(text))
            menu.tk_popup(event.x_root, event.y_root)
        widget.bind("<Button-3>", show_menu)

    def _bind_entry_context_menu(self, entry_widget):
        """Adds Cut, Copy, Paste, Select All right-click menu to entry field."""
        def show_menu(event):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="Cut", command=lambda: entry_widget.event_generate("<<Cut>>"))
            menu.add_command(label="Copy", command=lambda: entry_widget.event_generate("<<Copy>>"))
            menu.add_command(label="Paste", command=lambda: entry_widget.event_generate("<<Paste>>"))
            menu.add_separator()
            menu.add_command(label="Select All", command=lambda: entry_widget.event_generate("<<SelectAll>>"))
            menu.tk_popup(event.x_root, event.y_root)
        entry_widget.bind("<Button-3>", show_menu)
        if hasattr(entry_widget, "_entry"):
            entry_widget._entry.bind("<Button-3>", show_menu)

    def _add_user_message(self, text):
        """Displays user message on the right side of the chat log."""
        msg_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        msg_frame.pack(fill="x", pady=6, anchor="e")
        
        # User message bubble container
        bubble = ctk.CTkFrame(msg_frame, fg_color=("#1f6aa5", "#1f6aa5"), corner_radius=12)
        bubble.pack(side="right", padx=10)
        
        label = ctk.CTkLabel(
            bubble, 
            text=text, 
            text_color="white", 
            wraplength=450, 
            justify="left",
            font=ctk.CTkFont(size=13),
            padx=12,
            pady=8
        )
        label.pack(anchor="w")
        
        # Footer with copy button
        footer = ctk.CTkFrame(bubble, fg_color="transparent")
        footer.pack(anchor="e", padx=6, pady=(0, 4))
        
        copy_btn = ctk.CTkButton(
            footer,
            text="📋 Copy",
            width=55,
            height=20,
            fg_color="transparent",
            hover_color=("#195583", "#195583"),
            text_color="gray85",
            font=ctk.CTkFont(size=10),
            corner_radius=6
        )
        copy_btn.configure(command=lambda b=copy_btn, t=text: self._copy_to_clipboard(t, b))
        copy_btn.pack(side="right")
        
        # Enable right-click copy on bubble and label
        self._bind_copy_menu(label, text)
        self._bind_copy_menu(bubble, text)
        
        # Smooth scroll
        self.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _add_bot_message(self, text, score=None, suggestions=None):
        """Displays chatbot message on the left side of the chat log."""
        msg_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        msg_frame.pack(fill="x", pady=6, anchor="w")
        
        # Bot message bubble container
        bubble = ctk.CTkFrame(msg_frame, fg_color=("gray85", "gray25"), corner_radius=12)
        bubble.pack(side="left", padx=10, fill="both", expand=True)
        
        # Main text
        label = ctk.CTkLabel(
            bubble, 
            text=text, 
            text_color=("gray10", "gray90"), 
            wraplength=550, 
            justify="left",
            font=ctk.CTkFont(size=13),
            padx=12,
            pady=8
        )
        label.pack(anchor="w")
        
        # Footer for confidence score + copy button
        footer = ctk.CTkFrame(bubble, fg_color="transparent")
        footer.pack(fill="x", padx=12, pady=(0, 6))
        
        # Add metadata (like confidence score) if available
        if score is not None and score > 0.0:
            score_text = f"Confidence Match: {score*100:.1f}%"
            score_label = ctk.CTkLabel(
                footer, 
                text=score_text, 
                text_color="gray50", 
                font=ctk.CTkFont(size=10, slant="italic")
            )
            score_label.pack(side="left")
            
        copy_btn = ctk.CTkButton(
            footer,
            text="📋 Copy",
            width=55,
            height=20,
            fg_color="transparent",
            hover_color=("gray75", "gray35"),
            text_color=("gray20", "gray80"),
            font=ctk.CTkFont(size=10),
            corner_radius=6
        )
        copy_btn.configure(command=lambda b=copy_btn, t=text: self._copy_to_clipboard(t, b))
        copy_btn.pack(side="right")
        
        # Enable right-click copy on bubble and label
        self._bind_copy_menu(label, text)
        self._bind_copy_menu(bubble, text)
        
        # Add suggested options clickable labels directly inside message bubble if present
        if suggestions:
            sugg_box = ctk.CTkFrame(bubble, fg_color="transparent")
            sugg_box.pack(anchor="w", padx=12, pady=(0, 8), fill="x", expand=True)
            
            for idx, q_text in suggestions:
                s_btn = ctk.CTkButton(
                    sugg_box, 
                    text=f"❓ {q_text}", 
                    fg_color="transparent", 
                    text_color=("#1f6aa5", "#64b5f6"),
                    hover_color=("gray75", "gray35"),
                    anchor="w",
                    height=24,
                    font=ctk.CTkFont(size=12, underline=True),
                    command=lambda q=q_text: self._ask_custom_question(q)
                )
                s_btn.pack(fill="x", pady=2)
                
        # Smooth scroll
        self.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)

    def _on_language_changed(self, new_lang):
        """Called when user changes the language dropdown."""
        self.active_language = new_lang

    def _send_user_message(self):
        """Triggered when sending a message. Routes to FAQ engine or LLM based on active mode."""
        query = self.entry_field.get().strip()
        if not query:
            return
        
        # Prevent double-sends while LLM is streaming
        if self.is_llm_responding:
            return
            
        # Clear field
        self.entry_field.delete(0, tk.END)
        
        # Check for explicit language switch requests in query
        lower_q = query.lower()
        if any(p in lower_q for p in ["answer in english", "answer me in english", "in english please", "speak in english", "talk in english", "english me bolo", "english ma bolo", "english please"]):
            self.active_language = "English"
            if hasattr(self, 'lang_option'):
                self.lang_option.set("English")
        elif any(p in lower_q for p in ["in hinglish", "hinglish me", "hinglish ma", "speak in hinglish", "talk in hinglish", "hinglish bolo", "roman urdu"]):
            self.active_language = "Hinglish (Roman Urdu)"
            if hasattr(self, 'lang_option'):
                self.lang_option.set("Hinglish (Roman Urdu)")
        elif any(p in lower_q for p in ["in urdu", "urdu me", "urdu ma", "speak in urdu", "talk in urdu", "urdu bolo", "اردو میں"]):
            self.active_language = "Urdu (اردو)"
            if hasattr(self, 'lang_option'):
                self.lang_option.set("Urdu (اردو)")
        
        # Add user bubble
        self._add_user_message(query)
        
        if self.llm_mode_active and self.llm.base_url:
            # ---- LLM MODE: Stream response from Ollama ----
            self._send_llm_query(query)
        else:
            # ---- FAQ MODE: TF-IDF + Cosine Similarity ----
            ans, score, suggestions = self.engine.get_response(query, self.similarity_threshold)
            self.after(200, lambda: self._add_bot_message(ans, score, suggestions))
        
        # Refresh chips
        self.after(200, self._refresh_quick_questions)

    def _send_llm_query(self, query):
        """Sends query to Ollama LLM in a background thread with live streaming into chat."""
        self.is_llm_responding = True
        self.send_btn.configure(state="disabled", text="Thinking...")
        
        # Create a placeholder bot bubble that we will update with streaming tokens
        msg_frame = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        msg_frame.pack(fill="x", pady=6, anchor="w")
        
        bubble = ctk.CTkFrame(msg_frame, fg_color=("gray85", "gray25"), corner_radius=12)
        bubble.pack(side="left", padx=10, fill="both", expand=True)
        
        # Streaming text label inside bubble
        streaming_label = ctk.CTkLabel(
            bubble,
            text="Thinking...",
            text_color=("gray10", "gray90"),
            wraplength=550,
            justify="left",
            font=ctk.CTkFont(size=13),
            padx=12,
            pady=8
        )
        streaming_label.pack(anchor="w")
        
        # Accumulated response text (shared between threads via list)
        response_buffer = [""]
        
        def on_token(token_text):
            """Called from background thread for each token. Schedules GUI update."""
            response_buffer[0] += token_text
            current_text = response_buffer[0]
            self.after(0, lambda t=current_text: self._update_streaming_label(streaming_label, t))
        
        def on_complete(full_text, success, error):
            """Called when LLM response is fully received or on error."""
            def _finish():
                self.is_llm_responding = False
                self.send_btn.configure(state="normal", text="Send ➡️")
                
                final_text = response_buffer[0] if response_buffer[0] else full_text
                
                # Footer frame
                footer = ctk.CTkFrame(bubble, fg_color="transparent")
                footer.pack(fill="x", padx=12, pady=(0, 6))
                
                if not success:
                    streaming_label.configure(
                        text=f"LLM Error: {error}\n\n(Falling back to FAQ mode for this query)",
                        text_color="red"
                    )
                    # Fallback: try FAQ engine
                    ans, score, suggestions = self.engine.get_response(query, self.similarity_threshold)
                    self._add_bot_message(ans, score, suggestions)
                else:
                    # Add LLM badge with language tag
                    badge_text = f"Powered by LLM (Ollama) • {self.active_language}"
                    badge = ctk.CTkLabel(
                        footer,
                        text=badge_text,
                        text_color="gray50",
                        font=ctk.CTkFont(size=10, slant="italic")
                    )
                    badge.pack(side="left")
                    
                    # Copy button for LLM response
                    copy_btn = ctk.CTkButton(
                        footer,
                        text="📋 Copy",
                        width=55,
                        height=20,
                        fg_color="transparent",
                        hover_color=("gray75", "gray35"),
                        text_color=("gray20", "gray80"),
                        font=ctk.CTkFont(size=10),
                        corner_radius=6
                    )
                    copy_btn.configure(command=lambda b=copy_btn, t=final_text: self._copy_to_clipboard(t, b))
                    copy_btn.pack(side="right")
                    
                    # Enable right-click copy on streaming label and bubble
                    self._bind_copy_menu(streaming_label, lambda: response_buffer[0])
                    self._bind_copy_menu(bubble, lambda: response_buffer[0])
                
                # Scroll to bottom
                self.update_idletasks()
                self.chat_scroll._parent_canvas.yview_moveto(1.0)
            
            self.after(0, _finish)
        
        # Fire the background thread with active language
        self.llm.generate_response_threaded(
            query,
            self.engine.faqs,
            on_token_callback=on_token,
            on_complete_callback=on_complete,
            target_language=self.active_language
        )

    def _update_streaming_label(self, label, text):
        """Safely updates a streaming label's text on the main thread."""
        try:
            label.configure(text=text)
            self.update_idletasks()
            self.chat_scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _ask_suggestion(self, btn_idx):
        """Fires question from the quick suggestion chips."""
        question = self.suggestion_buttons[btn_idx].cget("text")
        if question:
            self.entry_field.delete(0, tk.END)
            self.entry_field.insert(0, question)
            self._send_user_message()

    def _ask_custom_question(self, question):
        """Fires question from inline bubble suggestion buttons."""
        self.entry_field.delete(0, tk.END)
        self.entry_field.insert(0, question)
        self._send_user_message()

    def _clear_chat_log(self):
        """Clears all conversation bubbles."""
        for widget in self.chat_scroll.winfo_children():
            widget.destroy()
        self._add_bot_message("Chat history cleared. How can I help you today?")
        self._refresh_quick_questions()

    def _refresh_quick_questions(self):
        """Pulls random questions from database to populate bottom suggestions."""
        if not self.engine.faqs:
            for btn in self.suggestion_buttons:
                btn.pack_forget()
            self.suggest_label.pack_forget()
            return
            
        self.suggest_label.pack(side="left", padx=5)
        # Select up to 3 random questions
        pool_size = min(len(self.engine.faqs), 3)
        selected_faqs = random.sample(self.engine.faqs, pool_size)
        
        for i in range(3):
            btn = self.suggestion_buttons[i]
            if i < len(selected_faqs):
                btn.configure(text=selected_faqs[i]['question'])
                btn.pack(side="left", padx=5)
            else:
                btn.pack_forget()

    # =========================================================================
    # FAQ DATABASE MANAGER TAB
    # =========================================================================
    def _setup_manager_frame(self):
        manager_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frames["manager"] = manager_frame
        
        # Divide into Left Column (List) and Right Column (Edit Form)
        manager_frame.grid_rowconfigure(0, weight=1)
        manager_frame.grid_columnconfigure(0, weight=6) # list (60%)
        manager_frame.grid_columnconfigure(1, weight=4) # form (40%)
        
        # --- LEFT PANEL (FAQ List) ---
        self.left_pane = ctk.CTkFrame(manager_frame, fg_color="transparent")
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(15, 5), pady=15)
        self.left_pane.grid_rowconfigure(1, weight=1)
        self.left_pane.grid_columnconfigure(0, weight=1)
        
        # Search & Count Row
        self.search_row = ctk.CTkFrame(self.left_pane, fg_color="transparent")
        self.search_row.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.search_row.grid_columnconfigure(0, weight=1)
        
        self.search_entry = ctk.CTkEntry(
            self.search_row, 
            placeholder_text="🔍 Search questions or answers in database...",
            height=35
        )
        self.search_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.search_entry.bind("<KeyRelease>", lambda e: self._refresh_manager_list())
        
        self.new_faq_btn = ctk.CTkButton(
            self.search_row, 
            text="+ Create New", 
            width=90, 
            height=35,
            command=self._prepare_create_form
        )
        self.new_faq_btn.grid(row=0, column=1)
        
        # FAQ Database List Container
        self.list_scroll = ctk.CTkScrollableFrame(self.left_pane, fg_color=("gray95", "gray20"))
        self.list_scroll.grid(row=1, column=0, sticky="nsew")
        self.list_scroll.grid_columnconfigure(0, weight=1)
        
        # --- RIGHT PANEL (Form) ---
        self.right_pane = ctk.CTkFrame(manager_frame, fg_color=("gray95", "gray20"), corner_radius=10)
        self.right_pane.grid(row=0, column=1, sticky="nsew", padx=(5, 15), pady=15)
        self.right_pane.grid_columnconfigure(0, weight=1)
        
        # Form Title
        self.form_title = ctk.CTkLabel(
            self.right_pane, 
            text="Add New FAQ", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.form_title.grid(row=0, column=0, padx=20, pady=(20, 15), sticky="w")
        
        # Question Label & Entry
        self.q_lbl = ctk.CTkLabel(self.right_pane, text="Question:", font=ctk.CTkFont(size=12, weight="bold"))
        self.q_lbl.grid(row=1, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.form_q_text = ctk.CTkTextbox(self.right_pane, height=80, corner_radius=6, border_width=1, border_color=("gray70", "gray40"))
        self.form_q_text.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # Answer Label & Entry
        self.a_lbl = ctk.CTkLabel(self.right_pane, text="Answer:", font=ctk.CTkFont(size=12, weight="bold"))
        self.a_lbl.grid(row=3, column=0, padx=20, pady=(5, 2), sticky="w")
        
        self.form_a_text = ctk.CTkTextbox(self.right_pane, height=180, corner_radius=6, border_width=1, border_color=("gray70", "gray40"))
        self.form_a_text.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        # Action Buttons Frame
        self.actions_frame = ctk.CTkFrame(self.right_pane, fg_color="transparent")
        self.actions_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        self.actions_frame.grid_columnconfigure(0, weight=1)
        self.actions_frame.grid_columnconfigure(1, weight=1)
        
        self.save_btn = ctk.CTkButton(
            self.actions_frame, 
            text="Save FAQ", 
            fg_color="green", 
            hover_color="darkgreen",
            command=self._submit_faq_form
        )
        self.save_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        
        self.delete_btn = ctk.CTkButton(
            self.actions_frame, 
            text="Delete", 
            fg_color="red", 
            hover_color="darkred",
            state="disabled",
            command=self._delete_selected_faq
        )
        self.delete_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")

    def _refresh_manager_list(self):
        """Clears and re-populates the FAQ list side using search filter query."""
        # Clear list
        for widget in self.list_scroll.winfo_children():
            widget.destroy()
            
        search_query = self.search_entry.get().strip().lower()
        
        # Populating items
        displayed_count = 0
        for idx, faq in enumerate(self.engine.faqs):
            q_text = faq['question']
            a_text = faq['answer']
            
            # Filter matches
            if search_query and (search_query not in q_text.lower() and search_query not in a_text.lower()):
                continue
                
            displayed_count += 1
            
            # Create list item frame card
            card = ctk.CTkFrame(
                self.list_scroll, 
                fg_color=("gray90", "gray25") if idx != self.selected_faq_idx else ("#1f6aa5", "#1f6aa5"),
                corner_radius=6,
                cursor="hand2"
            )
            card.pack(fill="x", pady=4, padx=5)
            
            # Pack inside card
            lbl_color = "white" if idx == self.selected_faq_idx else ("gray10", "gray90")
            lbl = ctk.CTkLabel(
                card, 
                text=q_text, 
                anchor="w", 
                justify="left", 
                text_color=lbl_color,
                wraplength=450,
                font=ctk.CTkFont(size=12, weight="bold" if idx == self.selected_faq_idx else "normal"),
                padx=10,
                pady=10
            )
            lbl.pack(fill="both", expand=True)
            
            # Bind click events to select item
            # Bind to both frame and label so clicking either selects
            card.bind("<Button-1>", lambda event, i=idx: self._select_faq_item(i))
            lbl.bind("<Button-1>", lambda event, i=idx: self._select_faq_item(i))

        # Show if empty
        if displayed_count == 0:
            empty_lbl = ctk.CTkLabel(self.list_scroll, text="No FAQs matching your query.", text_color="gray50")
            empty_lbl.pack(pady=20)

    def _select_faq_item(self, idx):
        """Loads selected FAQ into editing fields on the right side."""
        self.selected_faq_idx = idx
        faq = self.engine.faqs[idx]
        
        # Refresh highlighting in list
        self._refresh_manager_list()
        
        # Populate form
        self.form_title.configure(text="Edit FAQ Record")
        self.form_q_text.delete("1.0", tk.END)
        self.form_q_text.insert("1.0", faq['question'])
        self.form_q_text.configure(border_color=("gray70", "gray40"))
        
        self.form_a_text.delete("1.0", tk.END)
        self.form_a_text.insert("1.0", faq['answer'])
        self.form_a_text.configure(border_color=("gray70", "gray40"))
        
        # Enable buttons
        self.save_btn.configure(text="Update FAQ", fg_color="green")
        self.delete_btn.configure(state="normal")

    def _prepare_create_form(self):
        """Clears editing pane to allow creating a new FAQ."""
        self.selected_faq_idx = None
        self._refresh_manager_list()
        
        self.form_title.configure(text="Add New FAQ")
        self.form_q_text.delete("1.0", tk.END)
        self.form_a_text.delete("1.0", tk.END)
        
        self.save_btn.configure(text="Save FAQ", fg_color="#1f6aa5")
        self.delete_btn.configure(state="disabled")

    def _submit_faq_form(self):
        """Handles add/edit action when the user clicks save."""
        q = self.form_q_text.get("1.0", tk.END).strip()
        a = self.form_a_text.get("1.0", tk.END).strip()
        
        if not q or not a:
            messagebox.showwarning("Incomplete Fields", "Both Question and Answer fields must be filled.")
            return
            
        if self.selected_faq_idx is None:
            # CREATE Mode
            success, msg = self.engine.add_faq(q, a)
            if success:
                self._prepare_create_form()
                messagebox.showinfo("Success", "FAQ added to database and model retrained.")
            else:
                messagebox.showerror("Error", msg)
        else:
            # EDIT Mode
            success, msg = self.engine.edit_faq(self.selected_faq_idx, q, a)
            if success:
                messagebox.showinfo("Success", "FAQ updated and model retrained.")
            else:
                messagebox.showerror("Error", msg)
                
        self._refresh_manager_list()
        self._refresh_quick_questions()

    def _delete_selected_faq(self):
        """Handles deleting the currently selected FAQ."""
        if self.selected_faq_idx is None:
            return
            
        confirm = messagebox.askyesno(
            "Confirm Delete", 
            "Are you sure you want to delete this FAQ? This will immediately rebuild the matching models."
        )
        if not confirm:
            return
            
        success, msg = self.engine.delete_faq(self.selected_faq_idx)
        if success:
            messagebox.showinfo("Success", "FAQ deleted and database updated.")
            self._prepare_create_form()
            self._refresh_manager_list()
            self._refresh_quick_questions()
        else:
            messagebox.showerror("Error", msg)

    # =========================================================================
    # SETTINGS & INFO TAB
    # =========================================================================
    def _setup_settings_frame(self):
        settings_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.frames["settings"] = settings_frame
        
        # Grid layout (1 column, multi rows)
        settings_frame.grid_columnconfigure(0, weight=1)
        
        # --- TITLE ---
        title_lbl = ctk.CTkLabel(
            settings_frame, 
            text="Settings & System Diagnostics", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_lbl.pack(anchor="w", padx=30, pady=(25, 20))
        
        # Outer Card containing NLP matching configuration
        nlp_card = ctk.CTkFrame(settings_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        nlp_card.pack(fill="x", padx=30, pady=10)
        
        card_title = ctk.CTkLabel(nlp_card, text="NLP Matching Engine Settings", font=ctk.CTkFont(size=14, weight="bold"))
        card_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        # Slider Row for Similarity Threshold
        slider_frame = ctk.CTkFrame(nlp_card, fg_color="transparent")
        slider_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        self.threshold_label = ctk.CTkLabel(
            slider_frame, 
            text=f"Similarity Threshold: {self.similarity_threshold:.2f} (30%)", 
            font=ctk.CTkFont(size=12)
        )
        self.threshold_label.pack(anchor="w", pady=(0, 5))
        
        self.threshold_slider = ctk.CTkSlider(
            slider_frame, 
            from_=0.0, 
            to=1.0, 
            number_of_steps=100,
            command=self._on_threshold_change
        )
        self.threshold_slider.set(self.similarity_threshold)
        self.threshold_slider.pack(fill="x", pady=(0, 5))
        
        help_desc = ctk.CTkLabel(
            slider_frame, 
            text="Lower values match more easily but increase false positives. Higher values ensure precise matching but trigger suggestions more often.", 
            text_color="gray50",
            wraplength=700,
            justify="left",
            font=ctk.CTkFont(size=11, slant="italic")
        )
        help_desc.pack(anchor="w")
        
        # --- LLM CONFIGURATION CARD ---
        llm_card = ctk.CTkFrame(settings_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        llm_card.pack(fill="x", padx=30, pady=10)
        
        llm_title = ctk.CTkLabel(llm_card, text="LLM Server Configuration (Ollama)", font=ctk.CTkFont(size=14, weight="bold"))
        llm_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        llm_form = ctk.CTkFrame(llm_card, fg_color="transparent")
        llm_form.pack(fill="x", padx=20, pady=(0, 15))
        llm_form.grid_columnconfigure(1, weight=1)
        
        # Server URL
        url_lbl = ctk.CTkLabel(llm_form, text="Server URL:", font=ctk.CTkFont(size=12))
        url_lbl.grid(row=0, column=0, sticky="w", pady=5, padx=(0, 10))
        
        self.llm_url_entry = ctk.CTkEntry(
            llm_form,
            placeholder_text="e.g. https://your-tunnel.ngrok-free.dev",
            height=32
        )
        self.llm_url_entry.grid(row=0, column=1, sticky="ew", pady=5)
        if self.llm.base_url:
            self.llm_url_entry.insert(0, self.llm.base_url)
        
        # Model Name
        model_lbl = ctk.CTkLabel(llm_form, text="Model Name:", font=ctk.CTkFont(size=12))
        model_lbl.grid(row=1, column=0, sticky="w", pady=5, padx=(0, 10))
        
        self.llm_model_entry = ctk.CTkEntry(
            llm_form,
            placeholder_text="e.g. qwen2.5-coder:32b",
            height=32
        )
        self.llm_model_entry.grid(row=1, column=1, sticky="ew", pady=5)
        if self.llm.model_name:
            self.llm_model_entry.insert(0, self.llm.model_name)
        
        # Buttons row
        llm_btn_frame = ctk.CTkFrame(llm_form, fg_color="transparent")
        llm_btn_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.llm_save_btn = ctk.CTkButton(
            llm_btn_frame,
            text="Save Config",
            width=120,
            command=self._save_llm_config
        )
        self.llm_save_btn.pack(side="left", padx=(0, 10))
        
        self.llm_test_btn = ctk.CTkButton(
            llm_btn_frame,
            text="Test Connection",
            width=130,
            fg_color="#2fa572",
            hover_color="#1a7a50",
            command=self._test_llm_connection
        )
        self.llm_test_btn.pack(side="left", padx=(0, 10))
        
        self.llm_conn_status = ctk.CTkLabel(
            llm_btn_frame,
            text="Not configured",
            text_color="gray50",
            font=ctk.CTkFont(size=11)
        )
        self.llm_conn_status.pack(side="left", padx=10)
        
        # --- DATA MANAGEMENT CARD ---
        db_card = ctk.CTkFrame(settings_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        db_card.pack(fill="x", padx=30, pady=10)
        
        db_title = ctk.CTkLabel(db_card, text="Database Operations", font=ctk.CTkFont(size=14, weight="bold"))
        db_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        ops_frame = ctk.CTkFrame(db_card, fg_color="transparent")
        ops_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        # Select Profile Dropdown
        prof_lbl = ctk.CTkLabel(ops_frame, text="Load Built-in Dataset:", font=ctk.CTkFont(size=12))
        prof_lbl.grid(row=0, column=0, sticky="w", pady=5)
        
        self.profile_options = ctk.CTkOptionMenu(
            ops_frame, 
            values=["Overall / General Assistant", "E-commerce Support", "University Admissions"],
            command=self._on_dataset_profile_selected
        )
        self.profile_options.grid(row=0, column=1, sticky="w", padx=15, pady=5)
        
        # File operations
        io_lbl = ctk.CTkLabel(ops_frame, text="Custom File Operations:", font=ctk.CTkFont(size=12))
        io_lbl.grid(row=1, column=0, sticky="w", pady=5)
        
        io_btn_frame = ctk.CTkFrame(ops_frame, fg_color="transparent")
        io_btn_frame.grid(row=1, column=1, sticky="w", padx=15, pady=5)
        
        self.import_btn = ctk.CTkButton(
            io_btn_frame, 
            text="Import JSON/CSV", 
            width=140,
            command=self._import_custom_file
        )
        self.import_btn.pack(side="left", padx=(0, 10))
        
        self.export_btn = ctk.CTkButton(
            io_btn_frame, 
            text="Export Database", 
            width=140,
            command=self._export_database
        )
        self.export_btn.pack(side="left")
        
        # --- STATISTICS CARD ---
        stats_card = ctk.CTkFrame(settings_frame, fg_color=("gray95", "gray20"), corner_radius=8)
        stats_card.pack(fill="x", padx=30, pady=10)
        
        stats_title = ctk.CTkLabel(stats_card, text="Diagnostics & Environment Status", font=ctk.CTkFont(size=14, weight="bold"))
        stats_title.pack(anchor="w", padx=20, pady=(15, 10))
        
        self.stats_text_label = ctk.CTkLabel(
            stats_card, 
            text="Loading diagnostics...",
            justify="left", 
            anchor="w",
            font=ctk.CTkFont(size=12, family="Courier")
        )
        self.stats_text_label.pack(anchor="w", padx=20, pady=(0, 20))

    def _on_threshold_change(self, val):
        self.similarity_threshold = float(val)
        self.threshold_label.configure(text=f"Similarity Threshold: {self.similarity_threshold:.2f} ({int(self.similarity_threshold*100)}%)")

    def _on_dataset_profile_selected(self, val):
        """Loads pre-baked FAQ datasets from data folder."""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        if val == "Overall / General Assistant":
            target = os.path.join(data_dir, "general_faq.json")
        elif val == "E-commerce Support":
            target = os.path.join(data_dir, "ecommerce_faq.json")
        elif val == "University Admissions":
            target = os.path.join(data_dir, "university_faq.json")
        else:
            return
            
        self._load_faq_database(target)
        messagebox.showinfo("Dataset Loaded", f"Loaded profile: {val}")

    def _import_custom_file(self):
        """Prompts user to select a JSON or CSV file to import."""
        filepath = filedialog.askopenfilename(
            title="Import FAQ Dataset",
            filetypes=[("FAQ Files", "*.json;*.csv"), ("JSON Files", "*.json"), ("CSV Files", "*.csv")]
        )
        if not filepath:
            return
            
        self._load_faq_database(filepath)

    def _export_database(self):
        """Saves current memory FAQs to a local JSON file."""
        filepath = filedialog.asksaveasfilename(
            title="Export FAQ Dataset",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        if not filepath:
            return
            
        success, msg = self.engine.save_data(filepath)
        if success:
            messagebox.showinfo("Export Successful", msg)
        else:
            messagebox.showerror("Export Failed", msg)

    def _load_faq_database(self, filepath):
        """Wrapper to call load_data on engine and update state/UI."""
        success, msg = self.engine.load_data(filepath)
        if success:
            self.current_loaded_file = filepath
            # Reset selection in list
            self.selected_faq_idx = None
            # Update lists
            if hasattr(self, 'search_entry'):
                self.search_entry.delete(0, tk.END)
            self._prepare_create_form()
            self._refresh_quick_questions()
            self._update_settings_stats()
        else:
            messagebox.showerror("Loading Failed", msg)

    def _update_settings_stats(self):
        """Refreshes system indicators and counts in settings panel."""
        if not hasattr(self, 'stats_text_label'):
            return
            
        total_faqs = len(self.engine.faqs)
        status_scikit = "ENABLED (TF-IDF & Cosine)" if SKLEARN_AVAILABLE else "DISABLED (Fallback)"
        status_nltk = "ENABLED (Lemmatized)" if NLTK_AVAILABLE else "DISABLED (Basic)"
        
        llm_status = "Connected" if self.llm.is_connected else "Not connected"
        llm_url = self.llm.base_url if self.llm.base_url else "Not configured"
        llm_model = self.llm.model_name if self.llm.model_name else "Not set"
        mode_str = "LLM (Ollama)" if self.llm_mode_active else "FAQ (TF-IDF)"
        
        loaded_filename = os.path.basename(self.current_loaded_file)
        
        info = (
            f"Active Mode      : {mode_str}\n"
            f"Active FAQ File  : {loaded_filename}\n"
            f"Total FAQ Records: {total_faqs} entries\n"
            f"scikit-learn     : {status_scikit}\n"
            f"NLTK Library     : {status_nltk}\n"
            f"LLM Server       : {llm_url}\n"
            f"LLM Model        : {llm_model}\n"
            f"LLM Status       : {llm_status}\n"
            f"GUI Toolkit      : CustomTkinter v5.2.2"
        )
        self.stats_text_label.configure(text=info)

    def _save_llm_config(self):
        """Saves the LLM server URL and model name from Settings entries."""
        url = self.llm_url_entry.get().strip()
        model = self.llm_model_entry.get().strip()
        
        if not url:
            messagebox.showwarning("Missing URL", "Please enter the Ollama server URL (e.g., your ngrok address).")
            return
        if not model:
            messagebox.showwarning("Missing Model", "Please enter the model name (e.g., qwen2.5-coder:32b).")
            return
        
        self.llm.set_config(url, model)
        self.llm_conn_status.configure(text="Config saved!", text_color="#2fa572")
        self._update_settings_stats()
        messagebox.showinfo("Saved", f"LLM configuration saved.\nURL: {url}\nModel: {model}")

    def _test_llm_connection(self):
        """Tests connection to the Ollama server and updates status indicator."""
        url = self.llm_url_entry.get().strip()
        model = self.llm_model_entry.get().strip()
        
        if not url:
            messagebox.showwarning("Missing URL", "Enter the server URL first.")
            return
        
        # Temporarily set config for testing
        self.llm.set_config(url, model if model else self.llm.model_name)
        
        self.llm_conn_status.configure(text="Testing...", text_color="orange")
        self.update_idletasks()
        
        connected, msg = self.llm.check_connection()
        
        if connected:
            self.llm_conn_status.configure(text="Connected!", text_color="#2fa572")
            messagebox.showinfo("Connection Success", msg)
        else:
            self.llm_conn_status.configure(text="Failed", text_color="red")
            messagebox.showerror("Connection Failed", msg)
        
        self._update_settings_stats()
