import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import os
import random
import time
from faq_engine import FAQEngine, NLTK_AVAILABLE, SKLEARN_AVAILABLE

# Configure default customtkinter behavior
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class FAQChatbotGUI(ctk.CTk):
    def __init__(self, engine: FAQEngine, default_db_path: str):
        super().__init__()
        
        self.engine = engine
        self.default_db_path = default_db_path
        self.current_loaded_file = default_db_path
        self.similarity_threshold = 0.30
        
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
        self._add_bot_message("Hello! I am your FAQ Assistant. How can I help you today?")
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
        self.appearance_option.grid(row=6, column=0, padx=20, pady=(5, 20), sticky="ew")
        
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
        
        # Clear button
        self.clear_btn = ctk.CTkButton(
            self.chat_header, 
            text="Clear Chat", 
            width=80, 
            height=25, 
            fg_color="transparent", 
            border_width=1, 
            text_color=("gray10", "gray90"),
            command=self._clear_chat_log
        )
        self.clear_btn.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        
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
        
        self.send_btn = ctk.CTkButton(
            self.input_frame, 
            text="Send ➡️", 
            width=100, 
            height=45,
            corner_radius=8,
            command=self._send_user_message
        )
        self.send_btn.grid(row=0, column=1, sticky="ew")

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
        label.pack()
        
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
        
        # Add metadata (like confidence score) if available
        if score is not None and score > 0.0:
            score_text = f"Confidence Match: {score*100:.1f}%"
            score_label = ctk.CTkLabel(
                bubble, 
                text=score_text, 
                text_color="gray50", 
                font=ctk.CTkFont(size=10, slant="italic")
            )
            score_label.pack(anchor="w", padx=12, pady=(0, 5))
            
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

    def _send_user_message(self):
        """Triggered when sending a message. Performs similarity matching."""
        query = self.entry_field.get().strip()
        if not query:
            return
            
        # Clear field
        self.entry_field.delete(0, tk.END)
        
        # Add user bubble
        self._add_user_message(query)
        
        # Match responses in engine
        ans, score, suggestions = self.engine.get_response(query, self.similarity_threshold)
        
        # Simulate thinking response duration (very fast micro-delay for realistic UI feel)
        self.after(200, lambda: self._add_bot_message(ans, score, suggestions))
        
        # Refresh chips
        self.after(200, self._refresh_quick_questions)

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
            values=["E-commerce Support", "University Admissions"],
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
            text="📥 Import JSON/CSV", 
            width=140,
            command=self._import_custom_file
        )
        self.import_btn.pack(side="left", padx=(0, 10))
        
        self.export_btn = ctk.CTkButton(
            io_btn_frame, 
            text="📤 Export Database", 
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
        
        if val == "E-commerce Support":
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
        status_scikit = "✓ ENABLED (TF-IDF & Cosine)" if SKLEARN_AVAILABLE else "✗ DISABLED (Jaccard Overlap Fallback)"
        status_nltk = "✓ ENABLED (Lemmatized preprocessing)" if NLTK_AVAILABLE else "✗ DISABLED (Basic cleaning fallback)"
        
        loaded_filename = os.path.basename(self.current_loaded_file)
        
        info = (
            f"Active FAQ File  : {loaded_filename}\n"
            f"Total FAQ Records: {total_faqs} entries\n"
            f"scikit-learn     : {status_scikit}\n"
            f"NLTK Library     : {status_nltk}\n"
            f"GUI Toolkit      : CustomTkinter v5.2.2\n"
            f"Python Version   : 3.13.7"
        )
        self.stats_text_label.configure(text=info)
