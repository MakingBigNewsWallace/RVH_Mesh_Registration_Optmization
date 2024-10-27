import cv2
import os

def create_video_from_images(image_folder, output_video, frame_rate=10):
    images = []
    for root, dirs, files in os.walk(image_folder):
        for file in sorted(files):
            
            if file.endswith(".png") or file.endswith(".jpg"):
                images.append(os.path.join(root, file))
    # print(images)
    #put the last image to the first and remove the last image
    images.insert(0, images[-1])
    images.pop(-1)
    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get the dimensions
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    image_folder = "/home/wenbo/RVH_Mesh_Registration/opt_demo_file/debug_image"
    output_video = "/home/wenbo/RVH_Mesh_Registration/opt_demo_file/opt_iter_frame_593.mp4"
    create_video_from_images(image_folder, output_video)