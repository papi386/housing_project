import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from cleaning.pipeline import preprocess_pipeline

# ------------------------------
# MAIN APP CLASS
# ------------------------------
class ModelUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Price Prediction Interface")
        self.geometry("1100x700")
        self.csv_path = None
        self.df = None
        self.cleaned_df = None

        self._build_welcome_screen()

    # ------------------------------
    # SCREEN 1 – WELCOME / UPLOAD
    # ------------------------------
    def _build_welcome_screen(self):
        self.clear_screen()

        frame = tk.Frame(self)
        frame.pack(expand=True)

        tk.Label(frame, text="Welcome to the Price Prediction App", font=("Arial", 20, "bold")).pack(pady=20)
        tk.Label(frame, text="Upload a CSV file to start", font=("Arial", 12)).pack(pady=10)

        tk.Button(frame, text="Upload CSV", width=20, command=self.load_csv).pack(pady=10)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path:
            return

        self.csv_path = path
        self.df = pd.read_csv(path)
        self.cleaned_df=preprocess_pipeline(self.csv_path, "cleaned_data.csv")  # Run pipeline immediately
        self._build_cleaning_screen()

    # ------------------------------
    # SCREEN 2 – CLEANED DATA PREVIEW
    # ------------------------------
    def _build_cleaning_screen(self):
        self.clear_screen()

        tk.Label(self, text="Cleaned Dataset Preview", font=("Arial", 16, "bold")).pack(pady=10)

        text = tk.Text(self, height=15)
        text.pack(fill=tk.X, padx=20)
        text.insert(tk.END, self.cleaned_df.head().to_string())
        text.config(state=tk.DISABLED)

        tk.Button(self, text="Continue to Model Selection", command=self._build_model_screen).pack(pady=15)

    # ------------------------------
    # SCREEN 3 – MODEL SELECTION
    # ------------------------------
    def _build_model_screen(self):
        self.clear_screen()

        tk.Label(self, text="Choose a Model", font=("Arial", 16, "bold")).pack(pady=10)

        self.model_choice = tk.StringVar()
        models = ["XGBoost", "Random Forest", "Linear Regression"]
        ttk.Combobox(self, values=models, textvariable=self.model_choice, state="readonly").pack(pady=10)

        tk.Button(self, text="Test Model", command=self.run_model).pack(pady=20)

    # ------------------------------
    # SCREEN 4 – RESULTS
    # ------------------------------
    def run_model(self):
        if not self.model_choice.get():
            messagebox.showwarning("Warning", "Please select a model")
            return

        # Placeholder predictions
        self.cleaned_df['predicted_price'] = self.cleaned_df.iloc[:, 0].mean()

        self._build_results_screen()

    def _build_results_screen(self):
        self.clear_screen()

        tk.Label(self, text="Results", font=("Arial", 16, "bold")).pack(pady=10)

        # ---- TABLE BUTTON ----
        tk.Button(self, text="Show Predictions Table", command=self.show_table).pack(pady=5)

        # ---- METRICS ----
        metrics = ttk.Treeview(self, columns=("Metric", "Value"), show="headings")
        metrics.heading("Metric", text="Metric")
        metrics.heading("Value", text="Value")
        metrics.pack(pady=10)

        metrics.insert("", "end", values=("R2", "0.54"))
        metrics.insert("", "end", values=("MSE", "0.0036"))

        # ---- FIGURES ----
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(self.cleaned_df['predicted_price'], label="Predicted")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)

    def show_table(self):
        top = tk.Toplevel(self)
        top.title("Predictions Table")

        text = tk.Text(top)
        text.pack(expand=True, fill=tk.BOTH)
        text.insert(tk.END, self.cleaned_df.head(50).to_string())
        text.config(state=tk.DISABLED)

    # ------------------------------
    def clear_screen(self):
        for widget in self.winfo_children():
            widget.destroy()


# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app = ModelUI()
    app.mainloop()