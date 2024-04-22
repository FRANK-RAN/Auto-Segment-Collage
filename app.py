import sys
import os

# Add the parent directory to sys.path to access pipeline.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


from pipeline import process
import gradio as gr
import pdb

pic_dic = []

def update(input_cate):
    global pic_dic
    
    pic_dic = process(input_cate)
    avlaible_genres = list(pic_dic.keys())
    return gr.Dropdown(
            avlaible_genres,
            label="Available Genres", 
            interactive=True,
            info="Segementation and classification completed. Please choose the genre you would like to view."
        )
    
def get_select_cate(options, evt: gr.SelectData):
    pics = []
    for i in options:
        pics += pic_dic[i]
    gallery = gr.Gallery(
        pics, columns=[5], object_fit="contain", height=300, show_label=False, elem_id="gallery", label="Available Images", interactive=True
    )
    return gallery

def get_select_value(selected_gallery, evt: gr.SelectData):
    if selected_gallery:
        selected_gallery.append(evt.value['image']['path'])
    else:
        selected_gallery = gr.Gallery(
            [evt.value['image']['path']], columns=[5], object_fit="contain", height=200, show_label=False, elem_id="gallery", label="Selected Images", interactive=False
        )
    return selected_gallery

def collage():
    return "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg"

with gr.Blocks() as demo:    
    gr.Markdown('''
        # Welcome to Auto-Collage

        Welcome to **Auto-Collage**, 
        your go-to app for generating custom collages! 
        Whether you're an educator seeking engaging visual aids or an artist looking for inspiration, 
        our app provides a user-friendly platform to create unique collages of segmented images. 
        Powered by advanced AI models SAM and CLIP, we ensure high-quality, precision-segmented visuals. 
        Dive into our interactive interface below and start creating your personalized collage today!
                ''')
    with gr.Row():
        with gr.Group():
            inp = gr.Textbox(
                placeholder="Enter the path to your image dataset here, and click the `Process` button",
                label = "Dataset Path",
            )
            process_btn = gr.Button("Process")
        options = gr.Dropdown(
            [], 
            label="Available Genres", 
            interactive=True,
            info="Wait segementation and classification to start.",
            multiselect=True
        )
    with gr.Row():
        with gr.Column():
            gallery = gr.Gallery(
                [
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-116b5e92936b766b7fdfc242649337f7.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1163530ca19b5cebe1b002b8ec67b6fc.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1116395d6e6a6581eef8b8038f4c8e55.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1151ce9f4b2043de0d2e3b7826127998.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-116b5e92936b766b7fdfc242649337f7.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1163530ca19b5cebe1b002b8ec67b6fc.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-1116395d6e6a6581eef8b8038f4c8e55.jpg",
                "http://www.marketingtool.online/en/face-generator/img/faces/avatar-11319be65db395d0e8e6855d18ddcef0.jpg",
                ], columns=[5], object_fit="contain", show_label=False, elem_classes="gallery", height=300, label="Available Images", interactive=False
            )
            gr.Markdown('''
                Select the images you would like to include in your collage by clicking on them.
            ''')
            selected_gallery = gr.Gallery(
                [], columns=[5], object_fit="contain", show_label=False, elem_classes="gallery", height=200, label="Selected Images", interactive=False
            )
            collage_btn = gr.Button("Generate Collage")
        output = gr.Image(label="Collage")
    
    # Events
    gallery.select(get_select_value, inputs=selected_gallery, outputs=selected_gallery)
    process_btn.click(fn=update, inputs=inp, outputs=[options])
    options.select(get_select_cate, inputs=options, outputs=gallery)
    collage_btn.click(collage, None, output)

if __name__ == "__main__":
    demo.launch()