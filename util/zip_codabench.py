import os
import zipfile
import shutil

def zip_dir(dir_path, zip_path):
    """Zips the contents of a directory into the root of the zip file."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dir_path)
                zipf.write(file_path, arcname)

def main():
    base_dir = "codabench_bundle"
    master_zip_path = "codabench_bundle_v11.zip"
    
    print("Preparing Codabench assets (Hiding reference from UI)...")
    
    # 1. Ensure logo exists
    logo_src = "util/logo/logo.jpg"
    logo_dst = os.path.join(base_dir, "logo.jpg")
    if os.path.exists(logo_src):
        shutil.copy(logo_src, logo_dst)

    # 2. Re-create component zip files
    components = ['data', 'reference', 'scoring_program', 'starting_kit']
    for comp in components:
        dir_path = os.path.join(base_dir, comp)
        zip_path = os.path.join(base_dir, f"{comp}.zip")
        if os.path.exists(dir_path):
            zip_dir(dir_path, zip_path)

    # 3. Create the master bundle zip
    print(f"\nCreating master bundle: {master_zip_path}...")
    
    # Files that MUST be at the root for the server
    root_files = [
        "competition.yaml", "overview.md", "data.md", "evaluation.md", "terms.md", "logo.jpg",
        "data.zip", "reference.zip", "scoring_program.zip", "starting_kit.zip"
    ]
    
    # Files that should be in 'files/' for the public UI (Excluding reference.zip)
    public_files = [
        "data.zip", "scoring_program.zip", "starting_kit.zip"
    ]
    
    with zipfile.ZipFile(master_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add root files (All of them, server needs reference.zip)
        for file in root_files:
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                zipf.write(file_path, file)
        
        # Add public files to files/ folder (Hidden from public)
        for file in public_files:
            file_path = os.path.join(base_dir, file)
            if os.path.exists(file_path):
                zipf.write(file_path, f"files/{file}")
                
    print(f"\nSUCCESS! Upload '{master_zip_path}' as a NEW competition.")
    print("Reference.zip is at the root for scoring, but hidden from the 'files/' folder.")

if __name__ == "__main__":
    main()
