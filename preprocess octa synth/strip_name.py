import os

folder = "/data/scavone/octa-synth-packed_bigger_inverted/raw"  # ← change this to your folder path

for filename in os.listdir(folder):
    if filename.endswith("_inv.png"):  # adjust if other extensions exist
        new_name = filename.replace("_inv.png", ".png")
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))
