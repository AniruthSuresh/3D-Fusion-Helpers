import pathlib
import matplotlib
# Use 'Agg' backend for environments without a GUI (servers)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image

def summarize_dataset(base_dir, output_folder):
    base_path = pathlib.Path(base_dir)
    out_path = pathlib.Path(output_folder)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Processing iterations in {base_dir}...")

    for i in range(1, 51):
        folder_name = f"iter_{i:04d}"
        rgb_dir = base_path / folder_name / "third_person" / "rgb"
        
        if not rgb_dir.exists():
            print(f"⚠️ Warning: {rgb_dir} not found. Skipping.")
            continue

        # Gather and sort all png files
        images = sorted(list(rgb_dir.glob("*.png")), key=lambda x: x.name)
        
        if len(images) >= 2:
            img_first = Image.open(images[0])
            img_last = Image.open(images[-1])
            
            # Create the subplot layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            ax1.imshow(img_first)
            ax1.set_title(f"First: {images[0].name}")
            ax1.axis('off')
            
            ax2.imshow(img_last)
            ax2.set_title(f"Last: {images[-1].name}")
            ax2.axis('off')
            
            plt.suptitle(f"Iteration {i:04d} Summary", fontsize=16)
            
            # Save to disk
            save_fn = out_path / f"summary_iter_{i:04d}.png"
            plt.savefig(save_fn)
            plt.close(fig)  # Free up memory
            print(f"✅ Saved summary for {folder_name}")
        else:
            print(f"❓ Iteration {i:04d} has fewer than 2 images.")

    print(f"\n✨ Done! Check the '{output_folder}' directory for your results.")

# Run the function
summarize_dataset("dataset_randomized_initial_arm", "iteration_summaries")