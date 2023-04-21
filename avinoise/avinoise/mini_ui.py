# import tkinter as tk
# from tkinter import filedialog, messagebox
# import os


# def select_files():
#     files = filedialog.askopenfilenames()
#     file_paths.extend(list(files))
#     update_info_label()


# def select_folder():
#     clear_selections()
#     folder = filedialog.askdirectory()
#     for dirpath, _, filenames in os.walk(folder):
#         for filename in filenames:
#             file_path = os.path.join(dirpath, filename)
#             if file_path.lower().endswith('.mp3') or file_path.lower().endswith('.wav'):
#                 file_paths.append(file_path)
#     update_info_label()


# def update_info_label():
#     num_files = len(file_paths)
#     info_label.config(text=f"Selected files: {num_files}")


# def clear_selections():
#     global file_paths
#     file_paths = []
#     update_info_label()


# def save_file():
#     if len(file_paths) == 0:
#         messagebox.showerror("Error", "No files selected for analysis.")
#     else:
#         global filename
#         filename = ""
#         filename = filedialog.asksaveasfilename(defaultextension=".csv",
#                                                 initialfile="untitled.csv")
#         if not filename.endswith(".csv") and filename != "":
#             filename += ".csv"
#             textfield.delete(0, tk.END)
#             textfield.insert(0, filename)


# root = tk.Tk()
# # root.geometry("170x110")  # set window size
# root.title("AVINOISE")  # set window title

# file_paths = []

# info_label = tk.Label(root, text="")
# info_label.pack(side=tk.BOTTOM)
# update_info_label()

# select_files_button = tk.Button(root,
#                                 text="Select Files",
#                                 width=20,
#                                 command=select_files)
# select_files_button.pack(side=tk.TOP)

# select_folder_button = tk.Button(root,
#                                  text="Select Folder",
#                                  width=20,
#                                  command=select_folder)
# select_folder_button.pack()

# textfield = tk.Entry(root, width=50)
# textfield.pack()

# save_button = tk.Button(root, text="Export", command=save_file)
# save_button.pack()

# root.mainloop()

import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.title("AVINOISE")

# Frame to hold the left column
left_frame = tk.Frame(root)
left_frame.pack(side="left", fill="both", expand=True)

# Label for the left column
left_label = tk.Label(left_frame, text="Imported Files")
left_label.pack(pady=(10, 0))

# Text widget for the list of imported files
left_text = tk.Text(left_frame, wrap="word", font=("Arial", 12))
left_text.pack(fill="both", expand=True, padx=(10, 0), pady=(0, 10))

# Scrollbar for the left column
left_scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=left_text.yview)
left_scrollbar.pack(side="right", fill="y")

# Configure the left text widget to use the scrollbar
left_text.config(yscrollcommand=left_scrollbar.set)

# Frame to hold the right column
right_frame = tk.Frame(root)
right_frame.pack(side="right", fill="both", expand=True)

# Label for the right column
right_label = tk.Label(right_frame, text="Export Path and Logs")
right_label.pack(pady=(10, 0))

# Text widget for the export path and logs
right_text = tk.Text(right_frame, wrap="word", font=("Arial", 10))
right_text.pack(fill="both", expand=True, padx=(10, 0), pady=(0, 10))

# Scrollbar for the right column
right_scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=right_text.yview)
right_scrollbar.pack(side="right", fill="y")

# Configure the right text widget to use the scrollbar
right_text.config(yscrollcommand=right_scrollbar.set)

# Button to browse for files
def browse_file():
    global import_file_names
    import_file_names = filedialog.askopenfilenames(filetypes=[("WAV Files", "*.wav"), ("All files", "*.*")])
    if import_file_names:
        left_text.config(state="normal")
        left_text.delete("1.0", tk.END)
        for name in import_file_names:
            if name.endswith('.wav'):
                left_text.insert(tk.END, f"{name}\n")
            else:
                messagebox.showerror("Error", "Invalid file extension. Please select a CSV file.")
        
        left_text.config(state="disabled")


browse_button = tk.Button(left_frame, text="Import", command=browse_file)
browse_button.pack(pady=(10, 0))

# Button to choose the export path
def choose_export_path():
    filename = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="untitled.csv")
    right_text.insert(tk.END, f"{filename}\n")

choose_button = tk.Button(right_frame, text="Choose Export Path", command=choose_export_path)
choose_button.pack(pady=(10, 0))

if __name__ == "__main__":
    root.mainloop()
