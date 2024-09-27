# to be replaced by interop_services
class FileManager:
    
    def __init__(
            self):
        self.file = None
    
    
    def open_file(
            self,
            file_path):
        self.file = open(
            file_path,
            "r")
    
    
    def close_file(
            self):
        self.file.close()
    
    
    def read_file(
            self):
        return self.file.read()
    
    
    def write_file(
            self,
            data):
        self.file.write(
            data)
