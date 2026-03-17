import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import json
import os

# ---------------- CONFIG ----------------
IMAGE_PATH = "SRUniversity.png"
OUTPUT_JSON = "locations.json"
# 0.35 fits a 2048 tall image perfectly inside an 800px high window
INITIAL_SCALE = 0.35 
ZOOM_FACTOR = 1.25
# ----------------------------------------

# Load image
img = Image.open(IMAGE_PATH)
ORIG_W, ORIG_H = img.size

scale = INITIAL_SCALE
locations = {}

if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        locations = json.load(f)

# ---------------- UI ----------------
root = tk.Tk()
root.title(f"Campus Map Annotator ({ORIG_W}x{ORIG_H})")
root.geometry("1400x800")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Canvas area
canvas_frame = tk.Frame(main_frame)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Changed cursor to crosshair for precision
canvas = tk.Canvas(canvas_frame, bg="black", cursor="crosshair")
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=canvas.xview)
canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)

vbar.pack(side=tk.RIGHT, fill=tk.Y)
hbar.pack(side=tk.BOTTOM, fill=tk.X)

# Right panel
panel = tk.Frame(main_frame, width=300, padx=10)
panel.pack(side=tk.RIGHT, fill=tk.Y)

tk.Label(panel, text="Live Coordinates", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))

coord_label = tk.Label(panel, text="X: -\nY: -", font=("Consolas", 11))
coord_label.pack(anchor="w", pady=5)

tk.Label(panel, text="Saved Locations", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 5))

listbox = tk.Listbox(panel, height=20, exportselection=False)
listbox.pack(fill=tk.BOTH, expand=True)

# ---------------- FUNCTIONS ----------------
def render_image():
    global tk_img
    resized = img.resize(
        (int(ORIG_W * scale), int(ORIG_H * scale)),
        Image.LANCZOS
    )
    tk_img = ImageTk.PhotoImage(resized)
    canvas.delete("all")
    canvas.create_image(0, 0, image=tk_img, anchor="nw")
    canvas.config(scrollregion=(0, 0, tk_img.width(), tk_img.height()))
    redraw_markers()

def redraw_markers():
    for name, data in locations.items():
        cx = data["x"] * scale
        cy = data["y"] * scale
        canvas.create_oval(cx-5, cy-5, cx+5, cy+5, fill="red", outline="white")
        canvas.create_text(cx+8, cy-8, text=name, fill="red", anchor="nw", font=("Arial", 10, "bold"))

def save_json():
    with open(OUTPUT_JSON, "w") as f:
        json.dump(locations, f, indent=4)

def update_list():
    listbox.delete(0, tk.END)
    # Sort alphabetically to make finding places easier
    for k in sorted(locations.keys()):
        listbox.insert(tk.END, k)

def delete_selected():
    selection = listbox.curselection()
    if not selection:
        messagebox.showwarning("Warning", "Select a location from the list to delete.")
        return
    
    name = listbox.get(selection[0])
    if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name}'?"):
        del locations[name]
        save_json()
        update_list()
        render_image()

# ---------------- EVENTS ----------------
def on_mouse_move(event):
    cx = canvas.canvasx(event.x)
    cy = canvas.canvasy(event.y)
    ix = int(cx / scale)
    iy = int(cy / scale)

    if 0 <= ix < ORIG_W and 0 <= iy < ORIG_H:
        coord_label.config(text=f"X: {ix}\nY: {iy}")
    else:
        coord_label.config(text="X: -\nY: -")

def on_click(event):
    cx = canvas.canvasx(event.x)
    cy = canvas.canvasy(event.y)

    ix = int(cx / scale)
    iy = int(cy / scale)

    if ix < 0 or iy < 0 or ix > ORIG_W or iy > ORIG_H:
        return

    name = simpledialog.askstring("Location Name", "Enter location name:")
    if not name:
        return
    tag = simpledialog.askstring("Tag", "Enter tag (optional):")

    locations[name.title()] = {
        "x": ix,
        "y": iy,
        "tag": tag
    }

    save_json()
    update_list()
    render_image()

def zoom_in(event=None):
    global scale
    scale *= ZOOM_FACTOR
    render_image()

def zoom_out(event=None):
    global scale
    scale /= ZOOM_FACTOR
    render_image()

def on_mousewheel(event):
    # Standard vertical scroll
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

def on_shift_mousewheel(event):
    # Standard horizontal scroll
    canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")

def on_ctrl_mousewheel(event):
    # Ctrl + Scroll to zoom
    if event.delta > 0:
        zoom_in()
    else:
        zoom_out()

def on_listbox_select(event):
    # Automatically pan the canvas to the selected location
    selection = listbox.curselection()
    if not selection: return
    
    name = listbox.get(selection[0])
    data = locations.get(name)
    if not data: return
    
    # Calculate fraction of the way across the canvas
    cx = data["x"] * scale
    cy = data["y"] * scale
    
    # Try to center the point in the current view
    view_w = canvas.winfo_width()
    view_h = canvas.winfo_height()
    
    frac_x = (cx - (view_w / 2)) / (ORIG_W * scale)
    frac_y = (cy - (view_h / 2)) / (ORIG_H * scale)
    
    canvas.xview_moveto(max(0, frac_x))
    canvas.yview_moveto(max(0, frac_y))

# Middle-mouse pan
def pan_start(event):
    canvas.scan_mark(event.x, event.y)

def pan_move(event):
    canvas.scan_dragto(event.x, event.y, gain=1)

# ---------------- BINDINGS ----------------
canvas.bind("<Motion>", on_mouse_move)
canvas.bind("<Button-1>", on_click)

canvas.bind_all("<MouseWheel>", on_mousewheel)
canvas.bind_all("<Shift-MouseWheel>", on_shift_mousewheel)
canvas.bind_all("<Control-MouseWheel>", on_ctrl_mousewheel)

canvas.bind("<ButtonPress-2>", pan_start)
canvas.bind("<B2-Motion>", pan_move)

listbox.bind("<<ListboxSelect>>", on_listbox_select)

# ---------------- CONTROLS ----------------
controls = tk.Frame(panel, pady=10)
controls.pack(fill=tk.X)

# Grid layout for buttons
tk.Button(controls, text="Zoom In (+)", command=zoom_in).pack(fill=tk.X, pady=2)
tk.Button(controls, text="Zoom Out (-)", command=zoom_out).pack(fill=tk.X, pady=2)
tk.Button(controls, text="Delete Selected", command=delete_selected, fg="red").pack(fill=tk.X, pady=(15, 2))

# ---------------- START ----------------
render_image()
update_list()
root.mainloop()