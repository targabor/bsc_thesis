import tkinter as tk
import concurrent.futures
from tkinter import filedialog
from tkinter import messagebox
from src.dlls import cpp_caller
import platform
import subprocess


def basic_median_ever_frame():
    try:
        if not kernel_entry.get():
            messagebox.showerror("Empty input", "The kernel size is empty, please enter a value")
        else:
            assert int(kernel_entry.get()) > 1, messagebox.showerror("Kernel size error", "Kernel size must be greater than one!")
            assert int(kernel_entry.get()) % 2 == 1, messagebox.showerror("Kernel must be odd")
            cpp_caller.call_simple_median_for_video_frame('', int(kernel_entry.get()), input_text.get(), output_text.get())
            messagebox.showinfo("Done", "Basic median filter ended.")
    except Exception as e:
        messagebox.showerror("There is an error!", str(e))


def two_pass_ever_frame():
    try:
        cpp_caller.call_two_pass_median_for_video_frame('', input_text.get(), output_text.get())
        messagebox.showinfo("Done", "Two pass median filter ended.")
    except Exception as e:
        messagebox.showerror("There is an error!", str(e))


def weighted_median_ever_frame():
    print("weighted_median_ever_frame")


def dir_w_median_ever_frame():
    print("dir_w_median_ever_frame")


def basic_median_cube():
    print("basic_median_cube")


def two_pass_cube():
    print("two_pass_cube")


def weighted_median_cube():
    print("weighted_median_cube")


def dir_w_median_cube():
    print("dir_w_median_cube")


def switch_example_v2(case):
    case_dict = {
        "Basic median every frame": lambda: basic_median_ever_frame(),
        "Two pass median every frame": lambda: two_pass_ever_frame(),
        "Weighted median every frame": lambda: weighted_median_ever_frame(),
        "Directional weighted median every frame": lambda: dir_w_median_ever_frame(),
        "Basic median cube": lambda: basic_median_cube(),
        "Two pass median cube": lambda: two_pass_cube(),
        "Weighted median cube": lambda: weighted_median_cube(),
        "Directional weighted median cube": lambda: dir_w_median_cube(),
    }
    case_dict.get(case.get(), lambda: print("Bad input"))()


def check_valid(P):
    if P.isdigit() or P == "":
        return True
    else:
        messagebox.showerror("Error", "Invalid input! Please enter a number.")
        return False


def on_input_button():
    input_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mkv;*.avi;*.mov")])
    input_text.config(state='normal')
    input_text.delete(0, tk.END)
    input_text.insert(0, input_path)
    input_text.config(state='readonly')


def on_output_button():
    output_path = filedialog.askdirectory()
    output_text.config(state='normal')
    output_text.delete(0, tk.END)
    output_text.insert(0, output_path)
    output_text.config(state='readonly')


def on_submit():
    input_video = input_text.get()
    output_folder = output_text.get()
    if input_video.strip() == "" or output_folder.strip() == "":
        messagebox.showerror("Empty input or output", "The input or the output is empty, check it")
    else:
        switch_example_v2(i_filter)


def on_r_button():
    print(i_filter.get())


root = tk.Tk()
root.title("Video Converter")

for i in range(2):
    root.columnconfigure(i, minsize=250)
    root.rowconfigure(i, minsize=20)

options_frame = tk.Frame(root)
options_frame.grid(row=0, column=0, columnspan=2, sticky="W")
i_filter = tk.StringVar()
i_filter.set("Basic median every frame")
for i, option in enumerate(["Basic median every frame", "Two pass median every frame", "Weighted median every frame",
                            "Directional weighted median every frame", "Basic median cube", "Two pass median cube",
                            "Weighted median cube", "Directional weighted median cube"]):
    tk.Radiobutton(options_frame, text=option, variable=i_filter, value=option, command=on_r_button).grid(row=i,
                                                                                                          column=0,
                                                                                                          sticky="W")

data_frame = tk.Frame(root)
data_frame.grid(row=0, column=1, sticky="W")
input_label = tk.Label(data_frame, text="Input video")
input_label.grid(row=0, column=1)
input_text = tk.Entry(data_frame, state='readonly', width=50)
input_text.grid(row=1, column=1)
input_button = tk.Button(data_frame, text="Select", command=on_input_button)
input_button.grid(row=1, column=3)

output_label = tk.Label(data_frame, text="Output folder")
output_label.grid(row=2, column=1)
output_text = tk.Entry(data_frame, state='readonly', width=50)
output_text.grid(row=3, column=1)
output_button = tk.Button(data_frame, text="Select", command=on_output_button)
output_button.grid(row=3, column=3)

submit_button = tk.Button(data_frame, text="Submit", command=on_submit)
submit_button.grid(row=4, column=2, columnspan=2, pady=10)

extra_frame = tk.Frame(root)
extra_frame.grid(row=1, column=1, sticky="W")

frames_label = tk.Label(extra_frame, text="Frames to watch:")
frames_label.grid(row=0, column=0, sticky="W")
frames_entry = tk.Entry(extra_frame, validate="key", validatecommand=(extra_frame.register(check_valid), '%P'))
frames_entry.grid(row=0, column=1, sticky="W")

threshold_label = tk.Label(extra_frame, text="Threshold:")
threshold_label.grid(row=1, column=0, sticky="W")
threshold_entry = tk.Entry(extra_frame, validate="key", validatecommand=(extra_frame.register(check_valid), '%P'))
threshold_entry.grid(row=1, column=1, sticky="W")

kernel_label = tk.Label(extra_frame, text="Kernel:")
kernel_label.grid(row=2, column=0, sticky="W")
kernel_entry = tk.Entry(extra_frame, validate="key", validatecommand=(extra_frame.register(check_valid), '%P'))
kernel_entry.grid(row=2, column=1, sticky="W")

weight_type = tk.StringVar(extra_frame)
weight_type.set("uniform")

options = ["uniform", "distance"]

dropdown_label = tk.Label(extra_frame, text="Select method:")
dropdown_label.grid(row=3, column=0, sticky="W")
dropdown = tk.OptionMenu(extra_frame, weight_type, *options)
dropdown.grid(row=3, column=1, sticky="W")

root.mainloop()
