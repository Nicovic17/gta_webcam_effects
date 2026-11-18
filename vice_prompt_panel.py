"""Live prompt editor; all Tk calls stay on the camera's main/UI thread."""
from dataclasses import dataclass
import time

from vice_local_settings import DEFAULT_PROMPT, MAX_PROMPT_LENGTH, normalize_prompt

EDIT_DELAY_SECONDS = .65


class PromptDraft:
    """Debounce edits without applying empty/invalid text or duplicate prompts."""
    def __init__(self, prompt=DEFAULT_PROMPT):
        self.applied = normalize_prompt(prompt)
        self.text, self.edited_at = self.applied, 0.

    def edit(self, text, now):
        if text != self.text:
            self.text, self.edited_at = text, now

    def take(self, now, *, force=False):
        if not force and now-self.edited_at < EDIT_DELAY_SECONDS:
            return None
        prompt = normalize_prompt(self.text)
        if prompt == self.applied:
            return None
        self.applied = prompt
        return prompt


@dataclass(frozen=True)
class PanelUpdate:
    prompt: str | None = None
    toggle: bool = False
    strength: int | None = None


class PromptPanel:
    def __init__(self, prompt=DEFAULT_PROMPT):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        try:
            self._build(tk, ttk, prompt)
        except Exception:
            self.root.destroy()
            raise

    def _build(self, tk, ttk, prompt):
        self.draft = PromptDraft(prompt)
        self.pending_prompt = self.pending_strength = None
        self.pending_toggle = self.closed = False
        self.state = (False, 2)
        root = self.root
        root.title('VICE Locale - Descrivi il tuo effetto')
        root.configure(bg='#141c28')
        root.geometry(f'660x390+{max(20, root.winfo_screenwidth()-690)}+70')
        root.minsize(560, 360)
        root.protocol('WM_DELETE_WINDOW', root.withdraw)
        content = tk.Frame(root, bg='#141c28', padx=20, pady=16)
        content.pack(fill='both', expand=True)
        tk.Label(content, text='Cosa vuoi diventare?', font=('Segoe UI', 18, 'bold'),
                 fg='#f4f6fa', bg='#141c28', anchor='w').pack(fill='x')
        tk.Label(content, text='Descrivi stile, personaggio o ambiente. Meglio un testo breve in inglese.',
                 font=('Segoe UI', 10), fg='#b8c7dd', bg='#141c28', anchor='w',
                 wraplength=600, justify='left').pack(fill='x', pady=(5, 10))
        self.text = tk.Text(content, height=5, wrap='word', undo=True,
            font=('Segoe UI', 11), bg='#202d40', fg='#f4f6fa', insertbackground='white',
            relief='flat', padx=10, pady=8)
        self.text.pack(fill='both', expand=True)
        self.text.insert('1.0', self.draft.applied)
        self.text.edit_modified(False)
        self.text.bind('<<Modified>>', self._edited)
        self.text.bind('<Return>', self._enter)
        tk.Label(content, text='Automatico dopo una pausa di 0,65 s. Invio: applica subito. Shift+Invio: nuova riga.',
                 font=('Segoe UI', 9), fg='#b8c7dd', bg='#141c28', anchor='w',
                 wraplength=600, justify='left').pack(fill='x', pady=(6, 6))
        actions = tk.Frame(content, bg='#141c28')
        actions.pack(fill='x')
        ttk.Button(actions, text='Applica', command=lambda: self._apply(force=True)).pack(side='left')
        ttk.Button(actions, text='Ripristina GTA', command=self._restore).pack(side='left', padx=6)
        self.toggle_button = ttk.Button(actions, text='Avvia effetto', command=self._toggle)
        self.toggle_button.pack(side='right')
        self.level = ttk.Combobox(actions, state='readonly', width=13,
                                 values=('1 - Leggera', '2 - Media', '3 - Forte'))
        self.level.current(1)
        self.level.pack(side='right', padx=8)
        self.level.bind('<<ComboboxSelected>>', self._level_changed)
        self.feedback = tk.StringVar(value='Prompt GTA iniziale. Premi Avvia effetto per iniziare.')
        tk.Label(content, textvariable=self.feedback, font=('Segoe UI', 10),
                 fg='#83e4bf', bg='#141c28', anchor='w', wraplength=600,
                 justify='left').pack(fill='x', pady=(9, 2))
        tk.Label(content, text='SPAZIO nell\'anteprima: ON/OFF. P: riapre questo pannello. Q: esci dall\'anteprima.',
                 font=('Segoe UI', 9), fg='#b8c7dd', bg='#141c28', anchor='w',
                 wraplength=600, justify='left').pack(fill='x')

    def _edited(self, event=None):
        if self.text.edit_modified():
            self.text.edit_modified(False)
            value = self.text.get('1.0', 'end-1c')
            self.draft.edit(value, time.monotonic())
            self.feedback.set(f'Modifica in corso... {len(value)}/{MAX_PROMPT_LENGTH} caratteri')

    def _apply(self, *, force=False, now=None):
        now = time.monotonic() if now is None else now
        self.draft.edit(self.text.get('1.0', 'end-1c'), now)
        try:
            prompt = self.draft.take(now, force=force)
        except ValueError as error:
            self.feedback.set(str(error))
            return
        if prompt is not None:
            self.pending_prompt = prompt
            self.feedback.set('Nuovo prompt impostato.' if self.state[0] else
                              'Prompt pronto. Premi Avvia effetto per iniziare.')

    def _enter(self, event):
        if event.state & 1:  # Shift+Return keeps a multiline text field.
            return None
        self._apply(force=True)
        return 'break'

    def _restore(self):
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', DEFAULT_PROMPT)
        self.text.edit_modified(False)
        self._apply(force=True)

    def _toggle(self):
        self._apply(force=True)
        self.pending_toggle = not self.pending_toggle

    def _level_changed(self, event=None):
        self.pending_strength = self.level.current()+1

    def poll(self):
        # Never run a nested mainloop or wait for GPU work from a Tk callback.
        self.root.update_idletasks()
        self.root.update()
        self._apply()
        update = PanelUpdate(self.pending_prompt, self.pending_toggle, self.pending_strength)
        self.pending_prompt = self.pending_strength = None
        self.pending_toggle = False
        return update

    def set_state(self, enabled, strength):
        if self.state != (enabled, strength):
            if enabled != self.state[0]:
                self.feedback.set('Effetto attivo. Modifica il testo per cambiare la trasformazione.'
                                  if enabled else 'Effetto fermo. Premi Avvia effetto per ripartire.')
            self.state = enabled, strength
            self.toggle_button.configure(text='Interrompi effetto' if enabled else 'Avvia effetto')
            self.level.current(strength-1)

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.text.focus_set()

    def close(self):
        if not self.closed:
            self.closed = True
            self.root.destroy()
