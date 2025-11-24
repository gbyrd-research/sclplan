import os
import re
import glob
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def parse_log_file(log_file_path):
    """
    Parses the log file to extract timestamps and corresponding actions.
    """
    actions = {}
    log_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - AI2ThorActionLogger - INFO - (Agent \d+ -> .+)"
    )

    with open(log_file_path, 'r') as file:
        actions = file.readlines()
    processed_actions = list()
    for action in actions:
        processed_actions.append(" ".join(action.split("INFO -")[1:]))
    return processed_actions

def sort_images_by_timestamp(folder_path):
    """
    Sorts image files in a folder based on the timestamp in their filenames.
    """

    image_filenames = glob.glob(os.path.join(folder_path, "*"))

    # Function to extract timestamp and sort
    def extract_timestamp(filename):
        # Split the filename to extract the timestamp part
        base_name = filename.split("/")[-1]  # Get 'img_2024-12-04_18:54:17.069949.png'
        timestamp_str = base_name.split("_")[1] + "_" + base_name.split("_")[2]  # Get '2024-12-04_18:54:17.069949.png'
        timestamp_str = timestamp_str.replace(".png", "")
        return datetime.strptime(timestamp_str, "%Y-%m-%d_%H:%M:%S.%f")  # Convert to datetime object

    # Sort filenames by extracted timestamp
    sorted_filenames = sorted(image_filenames, key=extract_timestamp)

    return sorted_filenames

def overlay_text_on_image(image_path, text):
    """
    Overlays text on the image.
    """
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    font_size = 20

    try:
        # Use a common font, modify path if a system-specific font is desired
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Add text at the top left corner
    text_position = (10, 10)
    text_color = (255, 255, 255, 255)  # White
    outline_color = (0, 0, 0, 255)  # Black for outline

    # Draw outline for better visibility
    for offset in [-1, 1]:
        draw.text((text_position[0] + offset, text_position[1]), text, font=font, fill=outline_color)
        draw.text((text_position[0], text_position[1] + offset), text, font=font, fill=outline_color)

    # Draw the actual text
    draw.text(text_position, text, font=font, fill=text_color)

    return img

def save_processed_images(image_paths, actions, folder_path):
    """
    Saves images with superimposed actions as 'processed_img_{count}.png'.
    """
    for count, (action, image_path) in enumerate(zip(actions, image_paths)):
        processed_image = overlay_text_on_image(image_path, action)
        processed_image.save(os.path.join(folder_path, f"processed_img_{count}.png"))
    print("Processed images saved to the folder.")

def create_video(image_paths, actions, output_path, fps=1):
    """
    Creates an MP4 video from a list of image paths with 1 frame per second.
    """
    if not image_paths:
        print("No images found.")
        return

    # Read the first image to get the frame dimensions
    frame = cv2.imread(image_paths[0])
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for action, image_path in zip(actions, image_paths):
        overlaid_image = overlay_text_on_image(image_path, action)
        frame = cv2.cvtColor(np.array(overlaid_image), cv2.COLOR_RGBA2BGR)
        video.write(frame)

    video.release()
    print(f"Video saved to {output_path}")

def create_gif(image_paths, actions, output_path, duration=1000):
    """
    Creates a GIF from a list of image paths with 1 frame per second.
    """
    if not image_paths:
        print("No images found.")
        return

    images = []
    for action, image_path in zip(actions, image_paths):
        overlaid_image = overlay_text_on_image(image_path, action)
        images.append(overlaid_image)

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {output_path}")

def create_task_media(img_folder_path, log_file_path, output_vid_path, output_gif_path, actions):
    """Creates a video and gif of the agents completing a task in the
    environment.
    
    Args:
        img_folder_path (str): path to the saved, timestamped imgs from the sim
        agent_action_log_path (str): path to the actions.log file for the task
            that specifies in sequence which action was attempted
        output_vid_path (str): the desired path to the output video
        output_git_path (str): the desired path to the output gif

    Returns:
        None
    """

    # Parse the log file to get actions
    # actions = parse_log_file(log_file_path)

    # Sort images by timestamp
    image_paths = sort_images_by_timestamp(img_folder_path)

    # Save processed images with superimposed actions
    save_processed_images(image_paths, actions, img_folder_path)

    # Create MP4 video (1 frame per second)
    create_video(image_paths, actions, output_vid_path, fps=1)

    # Create GIF (1 frame per second)
    create_gif(image_paths, actions, output_gif_path, duration=1000)
