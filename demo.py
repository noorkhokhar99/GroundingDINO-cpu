import tkinter as tk
from tkinter import filedialog
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

# Define thresholds
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

def process_image():
    image_path = image_entry.get()
    text_prompt = text_entry.get()

    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_image.jpg", annotated_frame)

    result_label.config(text="Processing complete. Annotated image saved as annotated_image.jpg")

def browse_image():
    file_path = filedialog.askopenfilename()
    image_entry.delete(0, tk.END)
    image_entry.insert(0, file_path)

# GUI setup
app = tk.Tk()
app.title("Object Grounding GUI")

# Entry for image path
image_label = tk.Label(app, text="Image Path:")
image_label.pack()
image_entry = tk.Entry(app, width=50)
image_entry.pack()
browse_button = tk.Button(app, text="Browse", command=browse_image)
browse_button.pack()

# Entry for text prompt
text_label = tk.Label(app, text="Text Prompt:")
text_label.pack()
text_entry = tk.Entry(app, width=50)
text_entry.pack()

# Button to trigger processing
process_button = tk.Button(app, text="Process Image", command=process_image)
process_button.pack()

# Label for result
result_label = tk.Label(app, text="")
result_label.pack()

# Run the GUI
app.mainloop()
