# import os
# from megaparse import MegaParse

# model_dir = r"C:\Users\Kareen\.cache\onnxtr\models"
# files = os.listdir(model_dir)

# print("✅ Available Models in the Cache Directory:")
# print(files)


# from megaparse import MegaParse

# try:
#     megaparse = MegaParse()  # No model_path argument
#     print("✅ MegaParse successfully initialized!")
# except Exception as e:
#     print(f"❌ Error initializing MegaParse: {e}")


import shutil

# Check if pdftotext is found
poppler_path = shutil.which("pdftotext")
if poppler_path:
    print(f"✅ Poppler is installed and found at: {poppler_path}")
else:
    print("❌ Poppler is not found in PATH.")
