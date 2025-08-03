import imageio.v2 as imageio  # Use v2 for backward compatibility
import os

def make_gif(image_folder, output_path, fps=10):
    images = []
    filenames = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(".png")]
    )

    for filename in filenames:
        image_path = os.path.join(image_folder, filename)
        images.append(imageio.imread(image_path))

    imageio.mimsave(output_path, images, fps=fps)
    print(f"GIF saved to: {output_path}")

if __name__ == "__main__":
    image_folder = "./dreamerv3/logdir/test2-example/crafter/actual"
    output_gif = "./dreamerv3/logdir/gifs/episode_animation.gif"
    make_gif(image_folder, output_gif, fps=10)
