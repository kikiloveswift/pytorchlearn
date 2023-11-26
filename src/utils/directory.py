import os


class DocumentManager:

    @staticmethod
    def src_root_dir() -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_dir)
        return parent_directory
    @staticmethod
    def default_train_data_dir() -> str:
        return DocumentManager.src_root_dir() + '/data'


if __name__ == '__main__':
    print(DocumentManager.src_root_dir())
    print(DocumentManager.default_train_data_dir())