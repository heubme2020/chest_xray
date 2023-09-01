import pycdlib

def read_img_file(img_file_path):
    iso = pycdlib.PyCdlib()
    iso.open(img_file_path)

    for dir_record in iso.list_directory(iso_path='/'):
        if dir_record.is_dir():
            print(f"Directory: {dir_record.file_identifier()}")
        else:
            print(f"File: {dir_record.file_identifier()}")

    iso.close()

if __name__ == "__main__":
    img_file_path = 'jsrt/JPCNN080.IMG'
    read_img_file(img_file_path)