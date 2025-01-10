import streamlit as st
import requests
from PIL import Image
import os
import io
import matplotlib.pyplot as plt
import time

# Title for the app
st.header("Image Recognition App YOLOV5")


# Specify the folder where static images are stored
image_folder = "data/images/train2017"  # Replace with your image folder path

# List all image files in the folder
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(("jpg", "jpeg", "png"))])

# Display images in grid layout, each image is clickable
with st.sidebar:
    st.info("Select an Image")
    cols = st.columns(2)
    selected_image = None
    for i, image_name in enumerate(image_files):
        col = cols[i % 2].container(height=300, border=True)  # Distribute images in 4 columns
        img_path = os.path.join(image_folder, image_name)
        
        # Load the image for display as thumbnail
        image_thumb = Image.open(img_path)#.resize((150, 150))  # Resize for thumbnails
        if col.button(image_name, key=f"image_{i}"):
            selected_image = image_name
        col.image(image_thumb, caption=image_name, use_container_width=True)

# # Sidebar: Display thumbnails as clickable images
# st.sidebar.title("Select an Image")
# thumbnail_size = (500, 500)  # Size of the thumbnail images

# # Dictionary to store image paths and thumbnails
# thumbnails = {}

# # Create a sidebar layout with 3 columns
# cols = st.sidebar.columns(3)
# for i, image_file in enumerate(image_files):
#     pil_image = Image.open(os.path.join(image_folder, image_file))
#     #pil_image.thumbnail(thumbnail_size)
#     with cols[i % 3]:
#         if st.button(image_file, key=image_file):
#             st.image(pil_image, use_container_width=True)
#             selected_image_file = image_file

# # If a thumbnail is clicked, display the full image
if selected_image:
    with st.spinner("Detection in progress..."):
        time.sleep(0.5)
        # Load the full image
        image_path = os.path.join(image_folder, selected_image)
        pil_image = Image.open(image_path)

        # Display the selected image in the main area
        #st.image(pil_image, caption=f"Selected Image: {selected_image}", use_container_width=True)

        # Create a byte stream of the selected image for sending to FastAPI
        img_byte_array = io.BytesIO()
        pil_image.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)

        # Send the image to FastAPI for object detection
        files = {"file": ("image.png", img_byte_array, "image/png")}
        response = requests.post("http://127.0.0.1:8000/detect_objects/", files=files)

        # If the response is successful, display the result with bounding boxes
        if response.status_code == 200:
            detection_results = response.json()

            # Create a figure for bounding box visualization
            fig, ax = plt.subplots()
            ax.imshow(pil_image)

            # Plot bounding boxes
            for result in detection_results["results"]:
                x1, y1, x2, y2 = result["bbox"]
                class_name = result["class_name"]
                confidence = result["confidence"]

                # Draw the bounding box and label
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='green', linewidth=1))
                ax.text(x1 + 6, y1 - 7, f'{class_name} {confidence:.2f}', color='white', fontsize=6,
                        bbox=dict(facecolor='green', alpha=0.6, pad=3))

            # Main area: Display the image with bounding boxes
            st.success(f"Object detected Image: {selected_image}")
            #st.subheader(f"Object detected Image: {selected_image}")
            ax.axis('off')  # Hide axes for cleaner look
            st.pyplot(fig, use_container_width=True)
  
        else:
            st.error("Error detecting objects. Please try again.")


################################ okay ############################

# import streamlit as st
# import requests
# from PIL import Image
# import os
# import io
# import matplotlib.pyplot as plt

# # Title for the app
# st.title("Image Recognition App with Bounding Boxes")

# # Specify the folder where static images are stored
# image_folder = "data/images/train2017"  # Replace with your image folder path

# # List all image files in the folder
# image_files = [f for f in os.listdir(image_folder) if f.endswith(("jpg", "jpeg", "png"))]

# # Create a grid layout for displaying image thumbnails
# num_columns = 4  # Adjust the number of columns for thumbnails
# thumbnail_size = (150, 150)  # Size of the thumbnail images

# # Display thumbnails in a grid (each thumbnail is clickable)
# cols = st.columns(num_columns)
# for i, image_file in enumerate(image_files):
#     # Load image and create thumbnail
#     image_path = os.path.join(image_folder, image_file)
#     pil_image = Image.open(image_path)
#     pil_image.thumbnail(thumbnail_size)
    
#     # Display thumbnail as clickable image
#     with cols[i % num_columns]:
#         if st.button(image_file, key=image_file):  # Use the image filename as key
#             # When clicked, send image to FastAPI for prediction
#             img_byte_array = io.BytesIO()
#             pil_image.save(img_byte_array, format="PNG")
#             img_byte_array.seek(0)  # Seek to the beginning of the byte array

#             # Send the image to FastAPI endpoint for object detection
#             files = {"file": ("image.png", img_byte_array, "image/png")}
#             response = requests.post("http://127.0.0.1:8000/detect_objects/", files=files)

#             # If the response is successful, display the result with bounding boxes
#             if response.status_code == 200:
#                 detection_results = response.json()

#                 # Create a figure for bounding box visualization
#                 fig, ax = plt.subplots()
#                 ax.imshow(pil_image)

#                 # Plot bounding boxes
#                 for result in detection_results["results"]:
#                     x1, y1, x2, y2 = result["bbox"]
#                     class_name = result["class_name"]
#                     confidence = result["confidence"]

#                     # Draw the bounding box and label
#                     ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2))
#                     ax.text(x1 + 6, y1 - 7, f'{class_name} {confidence:.2f}', color='white', fontsize=8,
#                             bbox=dict(facecolor='blue', alpha=0.7))

#                 # Right side: Display the image with bounding boxes
#                 st.header("Image with Bounding Boxes:")
#                 ax.axis('off')  # Hide axes for cleaner look
#                 st.pyplot(fig)
#             else:
#                 st.error("Error detecting objects. Please try again.")
#         else:
#             # Show the thumbnail on the left side
#             st.image(pil_image, caption=f"Thumbnail: {image_file}")



















































# import streamlit as st
# import requests
# from PIL import Image
# import os
# import io
# import matplotlib.pyplot as plt

# # Title for the app
# st.title("Image Recognition App with Bounding Boxes")

# # Specify the folder where static images are stored
# image_folder = "data/images/train2017"  # Replace with your image folder path

# # List all image files in the folder
# image_files = [f for f in os.listdir(image_folder) if f.endswith(("jpg", "jpeg", "png"))]

# with st.sidebar:
#     # # Create two columns in Streamlit layout
#     # col1, col2 = st.columns([1, 1])


#     # Left side: Display the image selection dropdown and image
#     with st.container():
#         st.header("Select an Image:")
#         selected_image = st.selectbox("Choose an image from the list", image_files)

#         # Load and display the selected image
#         image_path = os.path.join(image_folder, selected_image)
#         pil_image = Image.open(image_path)
#         st.image(pil_image, caption=f"Selected Image: {selected_image}", use_container_width=True)

#         # Convert image to byte data for sending to FastAPI
#         img_byte_array = io.BytesIO()
#         pil_image.save(img_byte_array, format="PNG")
#         img_byte_array.seek(0)  # Seek back to the beginning of the byte array

#         # Button to send the image to FastAPI for detection
#         if st.button("Detect Objects"):
#             # Send the image to FastAPI endpoint for object detection
#             files = {"file": ("image.png", img_byte_array, "image/png")}
#             response = requests.post("http://127.0.0.1:8000/detect_objects/", files=files)

#             # If the response is successful, display the result with bounding boxes
#             if response.status_code == 200:
#                 detection_results = response.json()

#                 # Create a figure for bounding box visualization
#                 fig, ax = plt.subplots()
#                 ax.imshow(pil_image)

#                 # Plot bounding boxes
#                 for result in detection_results["results"]:
#                     x1, y1, x2, y2 = result["bbox"]
#                     class_name = result["class_name"]
#                     confidence = result["confidence"]

#                     # Draw the bounding box and label
#                     ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='blue', linewidth=2))
#                     ax.text(x1 + 6, y1 - 7, f'{class_name} {confidence:.2f}', color='white', fontsize=8,
#                             bbox=dict(facecolor='blue', alpha=0.7))
#                     ax.axis('off')  # Hide axes for cleaner look
# with st.container():
#     st.header("Image with Bounding Boxes:")
#     st.pyplot(fig)



# # Right side: Display the image with bounding boxes
# with col2:
#     st.header("Image with Bounding Boxes:")
#     ax.axis('off')  # Hide axes for cleaner look
#     st.pyplot(fig)
# else:
#     st.error("Error detecting objects. Please try again.")




















# import streamlit as st
# import requests
# from PIL import Image, ImageDraw
# import io
# import matplotlib.pyplot as plt

# st.title("Image Recognition App")

# # Upload image file
# uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#     # Convert the uploaded image to bytes
#     files = {"file": uploaded_file}

#     # Send POST request to the FastAPI endpoint
#     response = requests.post("http://127.0.0.1:8000/detect_objects/", files=files)

#     pil_image = Image.open(uploaded_file)

#     fig, ax = plt.subplots()
#     ax.imshow(pil_image)

#      # Process the response
#     if response.status_code == 200:
#         detection_results = response.json()
#         for result in detection_results["results"]:
#             x1, y1, x2, y2 = result["bbox"]
#             st.code(result["bbox"])
#             ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1, fill= False,edgecolor='blue', linewidth=1))
#             ax.text(x1 + 6, y1-7, f'{str(result["class_id"])} {result["confidence"]:.2f}', color='white', fontsize=6, bbox=dict(facecolor='blue', alpha=0.7))

#          # Hide the axes for a cleaner look
#         ax.axis('off')
#         st.pyplot(fig)